# Performance ledger

Every entry is **measured on the target machine** (i5-11300H / GTX 1650 Max-Q), with the
same command before and after, median of 3–5 runs. A change that did not beat run-to-run
variance was reverted, not kept — entries marked *rejected* are here so the idea is not
tried again.

## Baseline (before any work)

`python3 -m vipera` and a `simulate()` harness, medians:

| Workload | Before |
|---|---|
| `simulate` — 512 tokens, modelled routing (glm-5.3-flash) | 6231 ms |
| `simulate` — 545-token real trace, `lfu` (OLMoE) | 1131 ms |
| `sweep` glm-5.3-flash | 394 ms |
| `schedule` glm q4-experts | 106 ms |

`cProfile` named two functions, `_zipf_sample` (66%) and `ExpertCache._victim` (23%).

## Kept

| Idea | Baseline → Result | Verdict | Why |
|---|---|---|---|
| Build the Zipf weight table once per `_routing`, and sum it with `map` instead of a generator | modelled 6231 → 5176 ms | **kept** | The table was rebuilt per (token, layer) sample; the `map` sum is *numerically identical* (same order of addition) so every manifest is byte-identical. |
| `_victim`: scan `_order` for the first non-pinned key instead of building a candidate list and doing `k in candidates` | modelled 5176 → 2054 ms | **kept** | The old LRU path was an O(n²) scan. Manifests unchanged. |
| Don't maintain the O(n) recency list under `lfu`/`hot` | real trace 1131 → 52 ms (21.8×) | **kept** | `_order` is read only by LRU's `_victim`; `lfu`/`hot` paid `list.remove` on every access for a structure they never read. |

Cumulative: modelled path **3.1×**, real-trace path **21.8×**; `sweep` 394 → 174 ms,
`schedule --trace` ~9 ms. All 166 tests pass and every captured manifest/CLI output is
byte-identical to the baseline.

Guards: `tests/test_cache_policy.py::test_recency_list_is_only_maintained_for_lru` and
`::test_only_the_hot_policy_ever_pins` fail if the recency-list optimisation is reverted.

## Rejected

| Idea | Baseline → Result | Verdict | Why |
|---|---|---|---|
| Re-select experts with an O(k log n) weighted structure instead of the O(k·n) scan in `_zipf_sample` | — | **rejected** | It changes the order the weights are accumulated in, so the modelled routing (and every modelled number) changes. A faster wrong answer is not a win; the modelled generator stays exact. |
| Drop the redundant `key in self._order` membership scan in `_touch` | ~3% of one workload | **rejected** | Relies on an invariant that is not worth the crash if it ever breaks, for a gain inside the noise band. |
