# Measurement results

`tools/` holds the scripts that produce measurements, and `tools/*_results/`
holds the JSON/CSV/markdown outputs of those runs. Keep the directory layout
stable; the shell scripts below are the documented way to regenerate a set.

## Layout

| Directory | What it contains |
|---|---|
| `tools/l1a2_results/` | The L1a v3 imatrix measurement: 4 arms from Q8_0, PPL JSONs, `report_numbers.txt`, `run.log`. |
| `tools/l1a_results/` | The L1a v2 measurement (Q4_K_M -> Q4_K_M, byte-copy arms). Keep for provenance; it is superseded by `l1a2_results`. |
| `tools/l1a_results/REPORT.md` | The gate + 5-number report from the L1a v2 run. |
| `tools/l1a2_report.py` | Recomputes the 5 numbers from the `l1a2_results/*.json` files. |
| `tools/ppl_baseline.py` | The L5a PPL script: one model at a time, fixed eval/--ctx/--chunks, JSON output. |
| `tools/gpu_results/` | GPU `-ngl` sweep, KV dtype, threads, and 16K context measurements (+ `REPORT.md`). |
| `tools/server_results/` | `llama-bench`/`llama-server` A/B measurements for `spec_type` and MoE flags. |
| `tools/layer_results/` | Placement KV-cache measurements and `fa_cost` / `kv_*.json` inputs. |
| `tools/iq3_results/` | IQ3 decode spec measurements (if relevant to the current branch). |

## Single-measurement scripts

**L5a PPL (one model, fixed eval)**

```bash
python tools/l5a_perplexity.py run --model ~/.vipera/models/X.gguf \\
    --eval tools/eval/wikitext_sample.txt --label myrun
```

`tools/l1a2_run.sh` drives L1a v3 with resume: every PPL JSON that already has
`rc=0` is skipped, so rerunning the script only repeats the arms that failed.

**KL divergence vs F16/BF16 reference (L5b)**

```bash
python tools/l5b_kl.py --base base-f16.gguf --target q4km.gguf \\
    --eval tools/eval/wikitext_sample.txt --label q4km-vs-f16
```

The script probes `binary --help` for `--kl-divergence`; if the build does not
advertise it, run exits 1 with a rebuild hint instead of a misleading `None`.

## Resume rules

- A run JSON is only reused when it has `returncode == 0` and a non-`None`
  `ppl` (for `l5a_perplexity.py --resume`; for `l1a2_run.sh` the same check is
  done in the `PPL()` helper).
- A run JSON with `rc != 0` or `ppl == None` is always re-run.
- Do not edit a run JSON by hand and expect the reporting scripts to accept it:
  `l1a2_report.py` and `l1a_results/l1a_report.py` also hard-check the fixed
  parameters (`ctx`, `ngl`, `chunks` for L1a v2) and exit on drift.

## Privacy

None of these directories contains user-visible personal data. The models are
automatically excluded by `.gitignore` (`*.gguf`), and all measurement files are
regenerable from the source GGUF + eval text + `llama-perplexity` binary.
