#!/usr/bin/env python3
"""experiments/benchmark.py — the measurement harness for experiments/.

One variant, one model, one process invocation, one run, one log file. Every number the
harness reports comes from the measurement tool's own timing:

  * llama-bench  -> `avg_ts` / `stddev_ts` / `samples_ts` of the decode row (`-o json`)
  * llama-cli    -> `eval time = ... (MS ms per token, X tokens per second)`

A run that is killed, times out, or prints no parseable timing is recorded as a failure
with its exit status and an error tail. It is never recorded as a number. (plantodo.md
records why: an OOM kill used to surface as `n/a` with exit 0.)

Standing constraint from plantodo.md, enforced here: never run these benchmarks in
parallel on this 15 GiB / no-swap box — two concurrent runs each load a 10 GiB model and
the OOM killer takes one. The harness runs strictly sequentially and refuses to start
while another harness holds experiments/.lock.

Usage
    python3 experiments/benchmark.py run --variant baseline [--model KEY] [--runs N]
    python3 experiments/benchmark.py run --all
    python3 experiments/benchmark.py verify        # is the speculative variant lossless?
    python3 experiments/benchmark.py csv           # results/comparison.csv
    python3 experiments/benchmark.py charts        # results/charts/*.png
    python3 experiments/benchmark.py summary       # table on stdout
    python3 experiments/benchmark.py all           # everything above, in order
"""
import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = HERE / "results"
LOGS = HERE / "logs"
CHARTS = RESULTS / "charts"
LOCK = HERE / ".lock"
NVME = "nvme0n1"

VARIANTS = ["baseline", "flashattention", "kv-cache-quant", "speculative"]
LABELS = {                       # how a (variant, arm) cell is named in tables and charts
    ("baseline", "default"): "baseline",
    ("flashattention", "default"): "flashattention",
    ("kv-cache-quant", "default"): "kv-cache-quant",
    ("speculative", "control_no_draft"): "speculative control (llama-cli)",
    ("speculative", "speculative"): "speculative",
}


# --------------------------------------------------------------------------- config


def load_yaml(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_config():
    cfg = load_yaml(HERE / "config.yaml")
    cfg["_variant_defs"] = {
        "baseline": load_yaml(HERE / "baseline" / "control.yaml"),
        **{n: load_yaml(HERE / "variants" / n / "variant.yaml") for n in VARIANTS[1:]},
    }
    return cfg


# ------------------------------------------------------------------- /proc sampling


def diskstats_sectors(dev=NVME):
    try:
        with open("/proc/diskstats") as fh:
            for line in fh:
                p = line.split()
                if len(p) > 10 and p[2] == dev:
                    return int(p[5])          # sectors read
    except FileNotFoundError:
        pass
    return None


def sample_loop(pid, stop, out):
    """Poll /proc/<pid>/{stat,status} every 100 ms until the child exits."""
    out.update(majflt=0, vm_hwm_kib=0)
    while not stop.is_set():
        try:
            with open(f"/proc/{pid}/stat") as fh:
                fields = fh.read().rsplit(")", 1)[1].split()
                out["majflt"] = max(out["majflt"], int(fields[7]))
            with open(f"/proc/{pid}/status") as fh:
                for line in fh:
                    if line.startswith("VmHWM:"):
                        out["vm_hwm_kib"] = max(out["vm_hwm_kib"], int(line.split()[1]))
        except (FileNotFoundError, ProcessLookupError, IndexError):
            break
        time.sleep(0.1)


# ------------------------------------------------------------------ command building


def llm_bin(cfg, name):
    return os.path.join(os.path.expanduser(str(cfg["software"]["llama_cpp_dir"])), name)


def build_command(cfg, model_key, variant, arm=None, verbose=True):
    """Return (argv, label). `arm` picks the speculative variant's control arm."""
    m, meas = cfg["models"][model_key], cfg["measurement"]
    lv, vdef = meas["load_mode"][model_key], cfg["_variant_defs"][variant]
    model = str(REPO / m["path"])

    if vdef["runner"] == "llama-bench":
        argv = [llm_bin(cfg, "llama-bench"), "-m", model, "-p", "0",
                "-n", str(meas["n_gen"]), "-r", "1", "-t", str(meas["threads"]),
                "-o", "json", "-lm", lv] + list(vdef["flags"])
        return argv, f"{variant} {model_key} (llama-bench)"

    # `-st` (single turn) + stdin=/dev/null in run_one: this build has no -no-cnv flag,
    # and a run must never block waiting on stdin. `-v` is not for logging: it makes
    # llama-cli print the internal server's response JSON, whose `timings` object is the
    # exact decoder instrumentation (predicted_n / predicted_ms / predicted_per_second,
    # plus draft_n and draft_n_accepted when the speculative path is live).
    argv = [llm_bin(cfg, "llama-cli"), "-m", model, "-p", meas["prompt"],
            "-n", str(meas["n_gen"]), "-t", str(meas["threads"]), "-lm", lv,
            "-c", str(meas["ctx_size"]), "-s", str(meas["seed"]), "--temp", "0",
            "--ignore-eos", "-st", "-fa", "off", "-ctk", "f16", "-ctv", "f16"]
    if verbose:
        # -v is instrumentation, not logging: it makes llama-cli print the internal
        # server's response JSON, whose `timings` object is the exact decoder readout
        # (and, when the speculative path is live, its draft counters). It also splices
        # runtime log records into the transcript, so the correctness check runs without it.
        argv.append("-v")
    if arm == "speculative":
        argv += ["-md", str(REPO / vdef["draft"])] + list(vdef["draft_flags"])
    return argv, f"{variant}:{arm or 'control_no_draft'} {model_key} (llama-cli)"


# --------------------------------------------------------------------------- parsing


# This build (b1-f2cc728) prints a compact perf line and no longer prints
# `llama_perf_context_print: eval time = ...`:  [ Prompt: 304.1 t/s | Generation: 79.2 t/s ]
# Verified with and without -v/-lv 0. The detailed form is kept as a fallback for other builds.
RE_COMPACT = re.compile(r"\[\s*Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*"
                        r"Generation:\s*([0-9.]+)\s*t/s\s*\]")
RE_EVAL = re.compile(r"eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*runs?\s*"
                     r"\(\s*([0-9.]+)\s*ms per token,\s*([0-9.]+)\s*tokens per second\)")
RE_PROMPT_EVAL = re.compile(r"prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens?")
RE_ACCEPT = re.compile(r"acceptance rate\s*=\s*([0-9.]+)")
RE_NDRAFT = re.compile(r"n_draft\s*=\s*(\d+)")
RE_NACCEPT = re.compile(r"n_accept\s*=\s*(\d+)")


def parse_llama_bench(out):
    try:
        rows = json.loads(out)
    except json.JSONDecodeError:
        return None, "llama-bench did not print JSON"
    decode = [e for e in rows if e.get("n_gen", 0) > 0 and e.get("n_prompt", 0) == 0]
    if not decode:
        return None, f"no decode row in llama-bench JSON (n_rows={len(rows)})"
    e = decode[0]
    ts = e.get("avg_ts")
    if not ts:
        return None, "decode row has no avg_ts"
    return {
        "tok_s": round(float(ts), 2),
        "tok_s_std": round(float(e.get("stddev_ts") or 0.0), 2),
        "per_rep_tok_s": [round(float(x), 2) for x in (e.get("samples_ts") or [])],
        "ms_per_token": round(1000.0 / float(ts), 2),
        "n_threads": e.get("n_threads"),
        "flash_attn": e.get("flash_attn"),
        "type_k": e.get("type_k"),
        "type_v": e.get("type_v"),
        "n_gpu_layers": e.get("n_gpu_layers"),
        "model_size_mib": round((e.get("model_size") or 0) / 2**20, 1),
    }, None


def parse_llama_cli(out):
    """Decode tok/s from llama-cli. Compact line first, detailed printout as fallback."""
    c = RE_COMPACT.search(out)
    if c:
        res = {"tok_s": round(float(c.group(2)), 2),
               "prompt_tok_s": round(float(c.group(1)), 2),
               "perf_source": "compact line"}
        for key, rx in (("acceptance_rate_pct", RE_ACCEPT), ("n_draft", RE_NDRAFT),
                        ("n_accept", RE_NACCEPT)):
            mm = rx.search(out)
            if mm:
                res[key] = float(mm.group(1))
        return res, None
    m = RE_EVAL.search(out)
    if not m:
        return None, ("no decode timing in llama-cli output: neither the compact "
                      "`[ Prompt: X t/s | Generation: Y t/s ]` line nor the detailed "
                      "`eval time = ...` printout was present")
    ms, runs, ms_tok, tok_s = m.groups()
    res = {"tok_s": round(float(tok_s), 2), "ms_per_token": round(float(ms_tok), 2),
           "n_runs": int(runs), "eval_ms": round(float(ms), 1),
           "perf_source": "detailed printout"}
    p = RE_PROMPT_EVAL.search(out)
    if p:
        res["prompt_eval_ms"] = round(float(p.group(1)), 1)
        res["prompt_tokens"] = int(p.group(2))
    for key, rx in (("acceptance_rate_pct", RE_ACCEPT), ("n_draft", RE_NDRAFT),
                    ("n_accept", RE_NACCEPT)):
        mm = rx.search(out)
        if mm:
            res[key] = float(mm.group(1))
    return res, None


# ------------------------------------------------------------------------- execution


def run_one(argv, log_path, timeout_s, label):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sec0, t0 = diskstats_sectors(), time.time()
    sample = {}
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, text=True)
    stop = threading.Event()
    th = threading.Thread(target=sample_loop, args=(proc.pid, stop, sample), daemon=True)
    th.start()
    timed_out = False
    try:
        out, _ = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        out, _ = proc.communicate()
    stop.set()
    th.join(timeout=1.0)
    wall = time.time() - t0
    sec1 = diskstats_sectors()

    with open(log_path, "w") as fh:
        fh.write(f"# {label}\n# argv: {' '.join(argv)}\n# wall_s: {wall:.1f}\n\n")
        fh.write(out or "")

    rec = {"status": "ok", "exit_code": proc.returncode, "wall_s": round(wall, 1),
           "peak_rss_mib": round(sample.get("vm_hwm_kib", 0) / 1024.0),
           "majflt": sample.get("majflt", 0),
           "nvme_mib_read": (round((sec1 - sec0) * 512 / 2**20, 1)
                             if None not in (sec0, sec1) else None),
           "log": str(log_path.relative_to(REPO))}
    if timed_out:
        rec.update(status="timeout", error_tail=(out or "")[-300:])
        return rec
    if proc.returncode != 0:
        rec.update(status=f"exit_{proc.returncode}", error_tail=(out or "")[-300:])
        return rec
    rec["_stdout"] = out
    return rec


RE_SLOT_SPEC = re.compile(r'"speculative":(true|false)')
RE_SPEC_TYPES = re.compile(r'"speculative\.types":"([a-z_]+)"')
RE_NO_SPEC_IMPL = re.compile(r"no implementations specified for speculative decoding")
RE_CONTENT = re.compile(r'"content":"((?:[^"\\]|\\.)*)"')


RE_LOGLINE = re.compile(r"^\s*\x08?-?\d+\.\d{2}\.\d{3}\.\d{3}\s+[DIWE]\s")


def strip_log_noise(out):
    """Drop the `-v` runtime log records so the generated text can be read out.

    With -v the same stream carries log lines, so the transcript has to be de-interleaved
    by line shape before it can be compared between two runs.
    """
    keep = []
    for line in out.splitlines():
        if RE_LOGLINE.match(line) or "\x08" in line or line.startswith("libggml"):
            continue
        keep.append(line)
    return "\n".join(keep)


def extract_content(out):
    """Generated text of one run: de-interleaved transcript between the prompt echo and
    the perf line. ("" when it cannot be located.)"""
    text = strip_log_noise(out)
    if "\n> " not in text:
        return ""
    body = text.split("\n> ", 1)[-1].split("[ Prompt:", 1)[0]
    body = body.replace("[Start thinking]", "").replace("[End thinking]", "")
    return body.strip()


def parse_llama_cli_json(out):
    """Exact decoder instrumentation from the verbose llama-cli log.

    The build prints the internal server's response JSON. Its `timings` block is the same
    one a server client would read, and it is the only place the draft counters appear.
    """
    blocks, at = [], 0
    while True:
        i = out.find('"timings":', at)
        if i < 0:
            break
        j = i + len('"timings":')
        try:
            obj, consumed = json.JSONDecoder().raw_decode(out[j:])
        except json.JSONDecodeError:
            at = j
            continue
        if isinstance(obj, dict) and "predicted_per_second" in obj:
            blocks.append(obj)
        at = j + consumed
    if not blocks:
        return None, "no `timings` object with predicted_per_second in the verbose log"
    t = blocks[-1]
    res = {"tok_s": round(float(t["predicted_per_second"]), 2),
           "ms_per_token": round(float(t.get("predicted_per_token_ms", 0.0)), 2),
           "predicted_n": t.get("predicted_n"),
           "predicted_ms": round(float(t.get("predicted_ms", 0.0)), 1),
           "prompt_n": t.get("prompt_n"),
           "prompt_ms": round(float(t.get("prompt_ms", 0.0)), 1),
           "perf_source": "verbose timings JSON"}
    for key in ("draft_n", "draft_n_accepted"):
        if key in t:
            res[key] = t[key]
    if res.get("draft_n"):
        res["acceptance_rate_pct"] = round(
            100.0 * res.get("draft_n_accepted", 0) / res["draft_n"], 1)
    slot = RE_SLOT_SPEC.search(out)
    types = RE_SPEC_TYPES.search(out)
    res["slot_speculative"] = (slot.group(1) == "true") if slot else None
    res["spec_types"] = types.group(1) if types else None
    res["spec_impl_missing"] = bool(RE_NO_SPEC_IMPL.search(out))
    return res, None


def stats_for(runs):
    ok = [r for r in runs if r.get("status") == "ok" and r.get("tok_s")]
    vals = [r["tok_s"] for r in ok]
    out = {"ok": len(ok), "failed": len(runs) - len(ok), "n_runs": len(runs)}
    if not vals:
        return out
    out.update(median_tok_s=round(statistics.median(vals), 2),
               mean_tok_s=round(statistics.fmean(vals), 2),
               std_tok_s=round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0,
               min_tok_s=round(min(vals), 2), max_tok_s=round(max(vals), 2))
    out["cv_pct"] = (round(100.0 * out["std_tok_s"] / out["mean_tok_s"], 1)
                     if out["mean_tok_s"] else None)
    return out


def measure(cfg, model_key, variant, arm=None, runs=None):
    """Measure one (model, variant, arm) cell sequentially. Returns its result block."""
    runs = runs or cfg["measurement"]["runs"]
    vdef = cfg["_variant_defs"][variant]
    if model_key not in vdef["applicable_models"]:
        reason = (f"no draft model with the {model_key} tokenizer exists locally; "
                  f"{vdef.get('draft')} shares the Qwen3 tokenizer and pairs with "
                  f"Qwen3-30B-A3B, so this variant is measured on that model only"
                  if variant == "speculative"
                  else f"{variant} is not defined for {model_key}")
        return {"status": "not-applicable", "runner": vdef["runner"],
                "flags": list(vdef["flags"]), "reason": reason, "runs": []}
    argv, label = build_command(cfg, model_key, variant, arm)
    tag = variant if arm is None else f"{variant}_{arm}"
    block = {"model_path": cfg["models"][model_key]["path"],
             "model_bytes": cfg["models"][model_key]["bytes"],
             "load_mode": cfg["measurement"]["load_mode"][model_key],
             "runner": vdef["runner"],
             "flags": list(vdef["flags"]) + (list(vdef.get("draft_flags", []))
                                             if arm == "speculative" else []),
             "draft": vdef.get("draft") if arm == "speculative" else None,
             "command": " ".join(argv), "runs": []}
    # A discarded warm-up invocation per cell. Measured reason: without it, the first
    # recorded run of a cell is systematically slow (10.58 vs 13.96 tok/s on qwen3-4b,
    # 15.03 vs 23.37 on deepseek) because the model file is not yet in the page cache after
    # the previous cell read a different model. The warm-up is recorded for evidence and
    # excluded from the statistics.
    for w in range(1, cfg["measurement"].get("warmup_runs", 0) + 1):
        rec = run_one(argv, LOGS / f"{tag}_{model_key}_warmup{w}.log",
                      cfg["measurement"]["timeout_s"], f"{label} warm-up {w}")
        out = rec.pop("_stdout", None)
        if out is not None:
            metrics = (parse_llama_bench(out)[0] if vdef["runner"] == "llama-bench"
                       else (parse_llama_cli_json(out)[0] or parse_llama_cli(out)[0]))
            if metrics:
                rec.update(metrics)
        block.setdefault("warmup", []).append(
            {k: rec.get(k) for k in ("status", "tok_s", "wall_s", "peak_rss_mib",
                                     "nvme_mib_read", "log")})
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {tag} {model_key} "
              f"warm-up {w} (discarded) tok/s={rec.get('tok_s')} "
              f"wall={rec.get('wall_s')}s", flush=True)
    for i in range(1, runs + 1):
        log_path = LOGS / f"{tag}_{model_key}_run{i}.log"
        rec = run_one(argv, log_path, cfg["measurement"]["timeout_s"],
                      f"{label} run {i}")
        rec["run"] = i
        out = rec.pop("_stdout", None)
        if out is not None:
            if vdef["runner"] == "llama-bench":
                metrics, err = parse_llama_bench(out)
            else:
                metrics, err = parse_llama_cli_json(out)
                if metrics is None:
                    metrics, err = parse_llama_cli(out)
            if metrics is None:
                rec.update(status="unparsed", error_tail=err)
            else:
                rec.update(metrics)
        block["runs"].append(rec)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {tag} {model_key} "
              f"run {i}/{runs} status={rec['status']} tok/s={rec.get('tok_s')} "
              f"wall={rec['wall_s']}s", flush=True)
    block.update(stats_for(block["runs"]))
    # Did the lever actually engage? A number measured while the lever was off is not a
    # measurement of the lever, and must not be reported as one.
    runs_ok = [r for r in block["runs"] if r.get("status") == "ok"]
    if variant == "speculative" and runs_ok:
        engaged = any(r.get("slot_speculative") or r.get("draft_n") for r in runs_ok)
        block["lever_engaged"] = bool(engaged)
        if not engaged:
            note = ("draft model was loaded, but no speculative implementation was "
                    "available in this build: the runtime logs 'no implementations "
                    "specified for speculative decoding', the slot reports "
                    "speculative:false and the sampler list reports "
                    "speculative.types:'none' [measured, see "
                    "results/evidence_lever_and_oom.txt]. These tok/s rows therefore "
                    "measure the identical non-speculative decode path twice, not the "
                    "speculative lever.")
            block["lever_note"] = note
            block["status"] = "lever-not-engaged"
            print(f"  !! lever not engaged for {variant} {model_key}: {note}", flush=True)
    return block


# --------------------------------------------------------------------------- runner


def acquire_lock():
    if LOCK.exists():
        pid = LOCK.read_text().strip()
        if pid.isdigit() and Path(f"/proc/{pid}").exists():
            sys.exit(f"another harness is running (pid {pid}); refusing to run in "
                     f"parallel — plantodo.md: concurrent runs OOM-kill each other")
    LOCK.write_text(str(os.getpid()))


def release_lock():
    try:
        LOCK.unlink()
    except FileNotFoundError:
        pass


def result_path(variant):
    return RESULTS / ("baseline.json" if variant == "baseline"
                      else f"variant_{variant}.json")


def measure_variant(cfg, variant, models, runs):
    doc = {"variant": variant, "runner": cfg["_variant_defs"][variant]["runner"],
           "flags": cfg["_variant_defs"][variant]["flags"],
           "question": cfg["experiment"]["question"],
           "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "measurement": cfg["measurement"], "arms": {}}
    if variant == "speculative":
        # Same runner, same session: the control arm makes this variant comparable to
        # something measured under identical conditions rather than to a llama-bench row.
        for arm in ("control_no_draft", "speculative"):
            doc["arms"][arm] = {"models": {mk: measure(cfg, mk, variant, arm, runs)
                                           for mk in models}}
    else:
        doc["arms"] = {"default": {"models": {mk: measure(cfg, mk, variant, None, runs)
                                              for mk in models}}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = result_path(variant)
    # Merge rather than overwrite: measuring one model must not delete the cells already
    # measured for the other. Only the (arm, model) cells just measured are replaced.
    if path.exists():
        old = json.load(open(path))
        for arm, armdoc in doc["arms"].items():
            old.setdefault("arms", {}).setdefault(arm, {"models": {}})
            old["arms"][arm].setdefault("models", {}).update(armdoc["models"])
        old.update({k: doc[k] for k in ("variant", "runner", "flags", "question",
                                        "generated_at", "measurement")})
        doc = old
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2)
    print(f"wrote {path.relative_to(REPO)}", flush=True)
    return doc


def cmd_run(args):
    cfg = load_config()
    todo = VARIANTS if args.all else (args.variant or [])
    if not todo:
        sys.exit("give --variant NAME or --all")
    models = args.model or list(cfg["models"])
    acquire_lock()
    try:
        for variant in todo:
            if result_path(variant).exists() and not args.force:
                print(f"{result_path(variant).name} exists — skipping {variant} "
                      f"(use --force to redo)")
                continue
            measure_variant(cfg, variant, models, args.runs)
    finally:
        release_lock()


# ------------------------------------------------------------------------- verify


def verify(cfg, model_key, variant):
    """Speculative decoding must be lossless: same greedy tokens with and without draft."""
    arms = []
    for arm in (None, "speculative"):
        argv, label = build_command(cfg, model_key, variant, arm, verbose=False)
        rec = run_one(argv, LOGS / f"verify_{variant}_{arm or 'control'}_{model_key}.log",
                      cfg["measurement"]["timeout_s"], label)
        out = rec.pop("_stdout", "") or ""
        metrics = {}
        if out:
            metrics = parse_llama_cli_json(out)[0] or parse_llama_cli(out)[0] or {}
        arms.append({"arm": arm or "control_no_draft", "status": rec["status"],
                     "tok_s": metrics.get("tok_s"),
                     "predicted_n": metrics.get("predicted_n"),
                     "acceptance_rate_pct": metrics.get("acceptance_rate_pct"),
                     "slot_speculative": metrics.get("slot_speculative"),
                     "text": extract_content(out), "log": rec["log"]})
    identical = bool(arms[0]["text"]) and arms[0]["text"] == arms[1]["text"]
    # Engagement is read from the measured runs (which carry -v), not from these two runs:
    # with the lever off, identical text is expected and proves nothing.
    lever_on = False
    if result_path(variant).exists():
        doc = json.load(open(result_path(variant)))
        blk = doc.get("arms", {}).get("speculative", {}).get("models", {}).get(model_key, {})
        lever_on = bool(blk.get("lever_engaged"))
    if not lever_on:
        verdict = ("NOT CONCLUSIVE - the speculative path never engaged on this build, so "
                   "both arms ran the same decoder; identical text is expected")
    elif identical:
        verdict = "lossless - the draft reproduced the greedy output exactly"
    else:
        verdict = "OUTPUT CHANGED - the draft altered greedy decoding"
    result = {"variant": variant, "model": model_key,
              "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "check": "greedy output identical with and without the draft model",
              "lever_engaged": lever_on, "verdict": verdict,
              "identical": identical, "n_chars": len(arms[0]["text"]),
              "arms": arms}
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "verify_speculative.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"verify speculative on {model_key}: identical={identical} "
          f"({len(arms[0]['text'])} chars) -> verify_speculative.json", flush=True)
    return result


# ------------------------------------------------------------------------- reading


_CFG = None


def config():
    global _CFG
    if _CFG is None:
        _CFG = load_config()
    return _CFG


def control_of(variant, cfg=None):
    """(variant, arm) a variant's speedup is quoted against.

    baseline is the control for flashattention and for speculative's own llama-cli arm.
    kv-cache-quant is isolated against flashattention instead, because this build refuses
    quantized V cache with flash attention off (variant.yaml `control_arm`).
    """
    if variant == "speculative":
        return "speculative", "control_no_draft"
    ctrl = config()["_variant_defs"].get(variant, {}).get("control_arm", "baseline")
    return ctrl, "default"


def load_results():
    rows = []
    for variant in VARIANTS:
        path = result_path(variant)
        if not path.exists():
            continue
        doc = json.load(open(path))
        for arm, armdoc in doc.get("arms", {}).items():
            for model, block in armdoc.get("models", {}).items():
                rows.append({"variant": variant, "arm": arm, "model": model,
                             "block": block, "doc": doc})
    return rows


def label_of(row):
    return LABELS.get((row["variant"], row["arm"]),
                      f"{row['variant']}:{row['arm']}")


def find(rows, variant, arm, model):
    return next((r["block"] for r in rows if r["variant"] == variant
                 and r["arm"] == arm and r["model"] == model), None)


def measured(rows):
    return [r for r in rows if r["block"].get("median_tok_s")]


def perm_test(a, b, n=200000):
    """Two-sided permutation test on the difference of means, stdlib only.

    5 runs per cell is far too few for a normal approximation to be trusted; with 10 values
    the exact permutation distribution has only 252 splits, so the whole thing is computed
    (n is a safety cap for larger samples, sampling without replacement if ever needed)."""
    import itertools
    import random
    if not a or not b:
        return None
    obs = abs(statistics.fmean(a) - statistics.fmean(b))
    pool = list(a) + list(b)
    na = len(a)
    total = 0
    hits = 0
    if len(pool) <= 16:                      # exact
        for combo in itertools.combinations(range(len(pool)), na):
            pick = set(combo)
            x = [v for i, v in enumerate(pool) if i in pick]
            y = [v for i, v in enumerate(pool) if i not in pick]
            total += 1
            if abs(statistics.fmean(x) - statistics.fmean(y)) >= obs - 1e-12:
                hits += 1
    else:                                    # randomised
        rnd = random.Random(0)
        for _ in range(n):
            rnd.shuffle(pool)
            total += 1
            if abs(statistics.fmean(pool[:na]) - statistics.fmean(pool[na:])) >= obs - 1e-12:
                hits += 1
    return round(hits / total, 4) if total else None


def compare(r, rows):
    """(speedup, p_value) of a variant cell against its control cell."""
    cv, arm = control_of(r["variant"])
    ctrl = find(rows, cv, arm, r["model"])
    b = r["block"]
    if not ctrl or not b.get("median_tok_s") or not ctrl.get("median_tok_s"):
        return None, None
    speed = round(b["median_tok_s"] / ctrl["median_tok_s"], 3)
    vals_b = [x for _, x in per_run_tok_s(b)]
    vals_c = [x for _, x in per_run_tok_s(ctrl)]
    return speed, perm_test(vals_c, vals_b)


def cmd_summary(args):
    rows = load_results()
    if not rows:
        sys.exit("no results yet — run `benchmark.py run --all` first")
    print(f"{'model':<18}{'cell':<32}{'runner':<12}{'ok':>3}{'median':>9}{'std':>7}"
          f"{'cv%':>7}{'vs control':>12}{'perm p':>10}")
    for r in sorted(rows, key=lambda x: (x["model"], VARIANTS.index(x["variant"]),
                                         x["arm"])):
        b = r["block"]
        name = label_of(r)[:31]
        if b.get("status") == "not-applicable":
            print(f"{r['model']:<18}{name:<32}{'-':<12}{'-':>3}{'n/a':>9}{'':>7}"
                  f"{'':>7}{'n/a':>12}")
            continue
        if not b.get("median_tok_s"):
            print(f"{r['model']:<18}{name:<32}{b.get('runner',''):<12}"
                  f"{b.get('ok', 0):>3}{'FAILED':>9}{'':>7}{'':>7}"
                  f"{str(b.get('status', 'failed'))[:12]:>12}")
            continue
        speed, pval = compare(r, rows) if r["variant"] != "baseline" else (None, None)
        print(f"{r['model']:<18}{name:<32}{b.get('runner',''):<12}{b.get('ok',0):>3}"
              f"{b.get('median_tok_s',0):>9}{b.get('std_tok_s',0):>7}"
              f"{b.get('cv_pct',0):>7}"
              f"{(f'{speed:.3f}x' if speed else ''):>12}"
              f"{(f'p={pval}' if pval is not None else ''):>10}")


def cmd_csv(args):
    rows = load_results()
    if not rows:
        sys.exit("no results yet")
    fields = ["model", "cell", "variant", "arm", "runner", "control", "n_runs", "ok",
              "failed", "median_tok_s", "mean_tok_s", "std_tok_s", "cv_pct",
              "min_tok_s", "max_tok_s", "speedup_vs_control", "p_value_perm",
              "flags", "load_mode", "status", "lever_engaged", "note"]
    out = RESULTS / "comparison.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["model"], VARIANTS.index(x["variant"]),
                                             x["arm"])):
            b = r["block"]
            failed = not b.get("median_tok_s")
            speed, pval = (None, None) if (r["variant"] == "baseline" or failed) \
                else compare(r, rows)
            status = b.get("status", "measured")
            if failed and status in (None, "measured"):
                bad = [x.get("status") for x in b.get("runs", []) if x.get("status") != "ok"]
                status = bad[0] if bad else "failed"
            w.writerow({"model": r["model"], "cell": label_of(r), "variant": r["variant"],
                        "arm": r["arm"], "runner": b.get("runner", ""),
                        "control": ("" if r["variant"] == "baseline" or failed
                                    else ":".join(control_of(r["variant"]))),
                        "n_runs": b.get("n_runs", ""), "ok": b.get("ok", ""),
                        "failed": b.get("failed", ""),
                        "median_tok_s": b.get("median_tok_s", ""),
                        "mean_tok_s": b.get("mean_tok_s", ""),
                        "std_tok_s": b.get("std_tok_s", ""),
                        "cv_pct": b.get("cv_pct", ""), "min_tok_s": b.get("min_tok_s", ""),
                        "max_tok_s": b.get("max_tok_s", ""),
                        "speedup_vs_control": speed if speed is not None else "",
                        "p_value_perm": pval if pval is not None else "",
                        "flags": " ".join(b.get("flags", []) or []),
                        "load_mode": b.get("load_mode", ""),
                        "status": status,
                        "lever_engaged": b.get("lever_engaged", ""),
                        "note": b.get("reason", "") or b.get("lever_note", "") or (
                            "one lever isolated against the flashattention arm: this build "
                            "rejects quantized V cache with -fa off"
                            if r["variant"] == "kv-cache-quant" and not b.get("reason")
                            else "")})
    print(f"wrote {out.relative_to(REPO)}", flush=True)


# ------------------------------------------------------------------------- charts


def per_run_tok_s(block):
    return [(r["run"], r["tok_s"]) for r in block.get("runs", [])
            if r.get("status") == "ok" and r.get("tok_s")]


def cmd_charts(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = measured(load_results())
    if not rows:
        sys.exit("no measured results yet")
    CHARTS.mkdir(parents=True, exist_ok=True)
    models = sorted({r["model"] for r in rows})
    cells = [label_of(r) for r in rows]
    order = list(dict.fromkeys(cells))
    colors = {"baseline": "#444444", "flashattention": "#1f77b4",
              "kv-cache-quant": "#2ca02c", "speculative": "#d62728",
              "speculative control (llama-cli)": "#aaaaaa"}

    # (1) primary metric: median decode tok/s per cell, error bar = 1 std of the 5 runs
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / len(order)
    for i, cell in enumerate(order):
        xs, ys, es = [], [], []
        for j, model in enumerate(models):
            blk = next((r["block"] for r in rows if label_of(r) == cell
                        and r["model"] == model), None)
            if not blk or not blk.get("median_tok_s"):
                continue
            xs.append(j + i * width - 0.4 + width / 2)
            ys.append(blk["median_tok_s"])
            es.append(blk.get("std_tok_s") or 0.0)
        bars = ax.bar(xs, ys, width=width, yerr=es, capsize=3,
                      label=cell, color=colors.get(cell, "#999999"),
                      edgecolor="black", linewidth=0.4)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylabel("decode tok/s (median of 5 runs)")
    ax.set_title("Primary metric — decode throughput per variant\n"
                 "(error bars: 1 std over the 5 runs; vipera CPU-only build, "
                 "-t 4)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS / "primary_metric_bar.png", dpi=150)
    plt.close(fig)

    # (2) distribution: every run, not just the median
    fig, ax = plt.subplots(figsize=(10, 5))
    data, labels = [], []
    for model in models:
        for cell in order:
            blk = next((r["block"] for r in rows if label_of(r) == cell
                        and r["model"] == model), None)
            if blk and per_run_tok_s(blk):
                data.append([v for _, v in per_run_tok_s(blk)])
                labels.append(f"{model}\n{cell}")
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)
    for patch, lab in zip(bp["boxes"], labels):
        key = lab.split("\n")[1]
        patch.set_facecolor(colors.get(key, "#999999"))
        patch.set_alpha(0.6)
    for i, vals in enumerate(data, start=1):
        ax.plot([i] * len(vals), vals, "k.", markersize=3, alpha=0.7)
    ax.set_ylabel("decode tok/s (each dot = one run)")
    ax.set_title("Run-to-run distribution — 5 independent process invocations per cell")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS / "distribution_box.png", dpi=150)
    plt.close(fig)

    # (3) cross-validation over runs: does the effect hold in every run, or does one run
    #     carry it? Left: per-run speedup. Right: leave-one-run-out (jackknife) mean.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    allrows = load_results()
    for r in rows:
        if r["variant"] == "baseline":
            continue
        cv, arm = control_of(r["variant"])
        ctrl = find(allrows, cv, arm, r["model"])
        if not ctrl:
            continue
        cvals = dict(per_run_tok_s(ctrl))
        pairs = [(i, v / cvals[i]) for i, v in per_run_tok_s(r["block"]) if cvals.get(i)]
        if not pairs:
            continue
        ax.plot([p[0] for p in pairs], [p[1] for p in pairs], "o-",
                label=f"{r['model']} · {label_of(r)}", markersize=4)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("run index")
    ax.set_ylabel("speedup vs its own control (same run index)")
    ax.set_title("Per-run speedup — a conclusion that holds\nonly on average would "
                 "zig-zag across runs")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    jl_rows = []
    for r in rows:
        if r["variant"] == "baseline":
            continue
        cv, arm = control_of(r["variant"])
        ctrl = find(allrows, cv, arm, r["model"])
        if not ctrl:
            continue
        cvals = dict(per_run_tok_s(ctrl))
        pairs = [(i, v / cvals[i]) for i, v in per_run_tok_s(r["block"]) if cvals.get(i)]
        if len(pairs) < 2:
            continue
        ratios = [p[1] for p in pairs]
        jack = [statistics.fmean(ratios[:k] + ratios[k + 1:]) for k in range(len(ratios))]
        jl_rows.append((f"{r['model']}\n{label_of(r)}", jack))
    if jl_rows:
        ax.boxplot([v for _, v in jl_rows], tick_labels=[n for n, _ in jl_rows],
                   patch_artist=True, widths=0.5)
        for i, (_, vals) in enumerate(jl_rows, start=1):
            ax.plot([i] * len(vals), vals, "ko", markersize=4)
            ax.annotate(f"{statistics.fmean(vals):.3f}x", (i, statistics.fmean(vals)),
                        textcoords="offset points", xytext=(18, 0), fontsize=7)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("leave-one-run-out mean speedup")
    ax.set_title("Cross-validation over runs — dropping any single\nrun must not move the "
                 "conclusion")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS / "cross_validation.png", dpi=150)
    plt.close(fig)

    print(f"wrote {CHARTS.relative_to(REPO)}/primary_metric_bar.png, "
          f"distribution_box.png, cross_validation.png", flush=True)


# --------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="measure variants (sequential, one process at a time)")
    p.add_argument("--variant", action="append", choices=VARIANTS)
    p.add_argument("--all", action="store_true", help="baseline + the three variants")
    p.add_argument("--model", action="append", help="model key; repeatable")
    p.add_argument("--runs", type=int, default=None, help="override runs per cell")
    p.add_argument("--force", action="store_true", help="re-measure over an existing file")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("verify", help="speculative-decoding losslessness check")
    p.add_argument("--model", default=None)
    p.set_defaults(func=lambda a: verify(load_config(),
                                         a.model or load_config()["draft_model"]["pairs_with"],
                                         "speculative"))

    p = sub.add_parser("csv", help="write results/comparison.csv")
    p.set_defaults(func=cmd_csv)
    p = sub.add_parser("charts", help="write results/charts/*.png")
    p.set_defaults(func=cmd_charts)
    p = sub.add_parser("summary", help="print the comparison table")
    p.set_defaults(func=cmd_summary)

    p = sub.add_parser("all", help="run --all, then verify, csv, charts")
    p.add_argument("--runs", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=None)

    args = ap.parse_args()
    if args.cmd == "all":
        cfg = load_config()
        acquire_lock()
        try:
            for variant in VARIANTS:
                if result_path(variant).exists() and not args.force:
                    print(f"{result_path(variant).name} exists — skipping {variant}",
                          flush=True)
                    continue
                measure_variant(cfg, variant, list(cfg["models"]), args.runs)
        finally:
            release_lock()
        verify(cfg, cfg["draft_model"]["pairs_with"], "speculative")
        cmd_csv(args)
        cmd_charts(args)
        cmd_summary(args)
        return
    args.func(args)


if __name__ == "__main__":
    main()
