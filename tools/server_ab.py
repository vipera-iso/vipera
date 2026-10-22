#!/usr/bin/env python3
"""A/B a llama-server flag pair and report median decode speed and TTFT.

Used for the measurements in OPTIMIZE-techniques.md §4: the server applies
`--spec-type` and `--cache-reuse` itself, so the only honest comparison is two
child processes with one flag apart.

    PROMPT_FILE=$PWD/README.md REPS=3 python tools/server_ab.py base \\
        ~/.vipera/models/model.gguf 128 -- -ngl 17 -fa on -ctk q8_0 -ctv q8_0
    PROMPT_FILE=$PWD/README.md REPS=3 python tools/server_ab.py ngram \\
        ~/.vipera/models/model.gguf 128 -- -ngl 17 --spec-type ngram-map-k

Environment:
    REPS          requests per arm (default 2); the first is a cold prefill
    PROMPT_FILE   prompt text (default: a small code-rewrite prompt)
    TAIL          appended to every request after the first (multi-turn tests)
    LLAMA_SERVER  llama-server binary (default ~/llama-bin/llama-b11827/...)
    CUDA_LIB      directory holding libcudart (only needed by CUDA builds)
"""

import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BIN = os.environ.get("LLAMA_SERVER", os.path.expanduser("~/llama-bin/llama-b11827/llama-server"))
CUDA_LIB = os.environ.get("CUDA_LIB", os.path.expanduser("~/cuda/lib"))

# Default prompt = the one the recorded runs used (tools/server_results/
# ngram_prose_cache.jsonl): a file large enough that the model repeats much of
# it, which is where an n-gram drafter pays.
_DEFAULT_PROMPT = """Here is a file from my project:

```python
def load_config(path: Path) -> dict:
    \"\"\"Read a JSON config file, falling back to defaults.\"\"\"
    if not path.exists():
        return dict(DEFAULT_CONFIG)
    data = json.loads(path.read_text())
    return {**DEFAULT_CONFIG, **data}

def save_config(cfg: dict, path: Path) -> None:
    \"\"\"Write the config atomically.\"\"\"
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(cfg, indent=2))
    os.replace(tmp, path)

def quarantine(path: Path, exc: Exception) -> Path:
    \"\"\"Keep an unreadable file before anything overwrites it.\"\"\"
    backup = path.with_name(path.name + '.corrupt')
    shutil.copy2(path, backup)
    return backup
```

Rewrite the whole file with type hints added to every function signature and to
the local variables. Output only the full file.
"""


def post(port: int, payload: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def wait_ready(port: int, proc: subprocess.Popen, timeout: int = 300) -> float:
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited: {(proc.stderr.read() or '')[-1500:]}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
            return time.time() - start
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("server not ready")


def main() -> None:
    label, model, n_predict = sys.argv[1], sys.argv[2], int(sys.argv[3])
    rest = sys.argv[4:]
    if rest and rest[0] == "--":
        rest = rest[1:]
    port = 8099
    if "--port" in rest:
        i = rest.index("--port")
        port = int(rest[i + 1])
        del rest[i : i + 2]

    prompt_path = os.environ.get("PROMPT_FILE")
    prompt = Path(prompt_path).read_text() if prompt_path else _DEFAULT_PROMPT

    cmd = [
        BIN,
        "-m",
        model,
        "-c",
        "8192",
        "-np",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        *rest,
    ]
    env = dict(os.environ, LD_LIBRARY_PATH=CUDA_LIB)
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    try:
        load_s = wait_ready(port, proc)
        # Warm-up: pays the graph reservation so the first real request is not
        # biased by it, and leaves the prompt cache primed.
        post(port, {"prompt": "hi", "n_predict": 4, "temperature": 0})
        tail = os.environ.get("TAIL", "")
        rows = []
        for i in range(int(os.environ.get("REPS", "2"))):
            result = post(
                port,
                {
                    "prompt": prompt + (tail if i > 0 else ""),
                    "n_predict": n_predict,
                    "temperature": 0,
                    "cache_prompt": True,
                },
            )
            timings = result["timings"]
            rows.append((timings["predicted_per_second"], timings["prompt_ms"] / 1000.0))
        print(
            json.dumps(
                {
                    "label": label,
                    "load_s": round(load_s, 1),
                    "flags": rest,
                    "median_tps": round(statistics.median(r[0] for r in rows), 3),
                    "runs": [[round(r[0], 2), round(r[1], 2)] for r in rows],
                    "median_ttft_s": round(statistics.median(r[1] for r in rows), 2),
                }
            )
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
