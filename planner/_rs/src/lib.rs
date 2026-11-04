//! Island: CPython-exact MT19937 RNG, `_zipf_sample` -> `routing`, `ExpertCache`.
//!
//! The manifest must be byte-identical to the pure-Python path, so every
//! numeric choice here replicates CPython's `_randommodule.c` and `random.py`
//! exactly: 32-bit draws, the 53-bit `random()` shear, and every dict whose
//! insertion order can influence a result is an `IndexMap` (Python dicts are
//! insertion-ordered; `HashMap` is not).
//!
//! Only int seeds in [-2^63, 2^63) are accepted; `schedule.py` falls back to
//! the pure-Python path for anything else (`Random.sample` is ported
//! faithfully so the `skew <= 0` draw sequence matches, not re-seeded).

use std::collections::HashSet;

use indexmap::IndexMap;
use pyo3::exceptions::{PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

// -- MT19937, byte-exact with CPython's _randommodule.c ----------------------

struct Mt {
    mt: [u32; 624],
    mti: usize,
}

impl Mt {
    fn init_genrand(s: u32) -> [u32; 624] {
        let mut mt = [0u32; 624];
        mt[0] = s;
        for i in 1..624 {
            mt[i] = 1812433253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        mt
    }

    fn init_by_array(key: &[u32]) -> Self {
        // CPython's init_by_array starts from a full init_genrand(19650218)
        // state — not from a mostly-zero array with mt[0]=19650218.
        let mut mt = Self::init_genrand(19650218);
        let n = 624usize;
        let mut i = 1usize;
        let mut j = 0usize;
        let mut k = std::cmp::max(n, key.len());
        while k > 0 {
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1664525)))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= n {
                mt[0] = mt[n - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        let mut k = n - 1;
        while k > 0 {
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1566083941)))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= n {
                mt[0] = mt[n - 1];
                i = 1;
            }
            k -= 1;
        }
        mt[0] = 0x8000_0000;
        Mt { mt, mti: n }
    }

    fn genrand(&mut self) -> u32 {
        if self.mti >= 624 {
            // The regeneration is CPython's exact three-phase sequence the
            // first phase reads mt[kk+397] (still old), the second reads
            // mt[kk-227] (already new), and mt[623] is updated last from the
            // still-old mt[396]. A modular (i+397)\%624 loop would touch
            // mt[396] before mt[623] needs it and diverge from Python.
            const M: usize = 397;
            const MAG: [u32; 2] = [0, 0x9908_b0df];
            for kk in 0..(624 - M) {
                let y = (self.mt[kk] & 0x8000_0000) | (self.mt[kk + 1] & 0x7fff_ffff);
                self.mt[kk] = self.mt[kk + M] ^ (y >> 1) ^ MAG[(y & 1) as usize];
            }
            for kk in (624 - M)..(624 - 1) {
                let y = (self.mt[kk] & 0x8000_0000) | (self.mt[kk + 1] & 0x7fff_ffff);
                self.mt[kk] = self.mt[kk + M - 624] ^ (y >> 1) ^ MAG[(y & 1) as usize];
            }
            let y = (self.mt[623] & 0x8000_0000) | (self.mt[0] & 0x7fff_ffff);
            self.mt[623] = self.mt[M - 1] ^ (y >> 1) ^ MAG[(y & 1) as usize];
            self.mti = 0;
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }
}

fn seed_words(seed: i64) -> Vec<u32> {
    let a = seed.unsigned_abs();
    let bits = 64 - a.leading_zeros();
    let keyused = if bits == 0 { 1 } else { ((bits - 1) / 32 + 1) as usize };
    let mut words = vec![0u32; keyused];
    for (i, w) in words.iter_mut().enumerate() {
        *w = ((a >> (32 * i)) & 0xffff_ffff) as u32;
    }
    // CPython 3.11+_randommodule.c has NO nonzero-word fixup: seed=0 yields
    // key=[0] passed straight to init_by_array. (An older CPython had one,
    // which is why key=[0] must NOT be "corrected" to [1] here.)
    words
}

// -- RNG class (also the parity-test oracle for `random.Random`) -------------

#[pyclass]
pub struct RsRandom {
    mt: Mt,
}

#[pymethods]
impl RsRandom {
    #[new]
    #[pyo3(signature = (seed = 0))]
    fn new(seed: i64) -> Self {
        RsRandom { mt: Mt::init_by_array(&seed_words(seed)) }
    }

    /// Debug: raw 624 words + index, to diff against `Random.getstate()`.
    #[pyo3(name = "_state_words")]
    fn state_words_py(&self) -> (Vec<u32>, usize) {
        (self.mt.mt.to_vec(), self.mt.mti)
    }

    /// CPython `random()`: two 32-bit draws sheared into 53 bits, in the same
    /// arithmetic order `a * 2**26 + b` over `2**53`.
    fn random(&mut self) -> f64 {
        let a = (self.mt.genrand() >> 5) as f64;
        let b = (self.mt.genrand() >> 6) as f64;
        (a * 67108864.0 + b) / 9007199254740992.0
    }

    /// CPython `_getrandbits`: least-significant word generated first; a
    /// remainder top word loses its least significant bits (`r >>= 32 - rem`).
    fn getrandbits(&mut self, k: u32) -> PyResult<Py<PyAny>> {
        if k == 0 {
            return Python::with_gil(|py| {
                Ok(0u64.into_pyobject(py)?.into_any().unbind())
            });
        }
        if k > 64 {
            return Err(PyValueError::new_err("island getrandbits supports k <= 64 only"));
        }
        let words = ((k - 1) / 32 + 1) as usize;
        let mut value: u64 = 0;
        for i in 0..words {
            // CPython: word i draws at k' = k - 32*i; when k' < 32 the drawn
            // word keeps only its TOP k' bits (r >>= 32-k'). The shift lands
            // on the LAST word, never the first — a 48-bit call draws
            // (genrand() full, genrand() >> 16).
            let rem = k - (32 * i) as u32;
            let mut r = self.mt.genrand();
            if rem < 32 {
                r >>= 32 - rem; // Drop least significant bits
            }
            value |= (r as u64) << (32 * i);
        }
        Python::with_gil(|py| Ok(value.into_pyobject(py)?.into_any().unbind()))
    }

    /// CPython `Random.sample(range(n), k)` — both paths (pool and set), with
    /// the same `setsize` heuristic, so the draw sequence matches exactly.
    #[pyo3(name = "sample")]
    fn sample_py(&mut self, n: u64, k: u64) -> PyResult<Vec<u64>> {
        self.sample(n, k)
    }
}

impl RsRandom {
    fn getrandbits_u64(&mut self, k: u32) -> u64 {
        if k == 0 {
            return 0;
        }
        let words = ((k - 1) / 32 + 1) as usize;
        let mut value: u64 = 0;
        for i in 0..words {
            // Same as getrandbits(): shift applies to the last word only.
            let rem = k - (32 * i) as u32;
            let mut r = self.mt.genrand();
            if rem < 32 {
                r >>= 32 - rem;
            }
            value |= (r as u64) << (32 * i);
        }
        value
    }

    /// CPython `_randbelow_with_getrandbits`.]
    fn randbelow(&mut self, n: u64) -> PyResult<u64> {
        if n == 0 {
            return Err(PyValueError::new_err("_randbelow: n must be > 0"));
        }
        let k = 64 - n.leading_zeros();
        let mut r = self.getrandbits_u64(k);
        while r >= n {
            r = self.getrandbits_u64(k);
        }
        Ok(r)
    }

    fn sample(&mut self, n: u64, k: u64) -> PyResult<Vec<u64>> {
        if k > n {
            return Err(PyValueError::new_err(
                "Sample larger than population or is negative",
            ));
        }
        let mut result = vec![0u64; k as usize];
        if k == 0 {
            return Ok(result);
        }
        let mut setsize = 21.0f64;
        if k > 5 {
            // math.log(k * 3, 4) is log(k*3) / log(4); ceil; then 4 ** that.
            let l = (3.0 * k as f64).ln() / 4.0f64.ln();
            setsize += 4.0f64.powf(l.ceil());
        }
        if (n as f64) <= setsize {
            // Pool path: non-selected keyâs live at pool[0 : n-i].
            let mut pool: Vec<u64> = (0..n).collect();
            for i in 0..k as usize {
                let j = self.randbelow(n - i as u64)? as usize;
                result[i] = pool[j];
                pool[j] = pool[n as usize - i - 1];
            }
        } else {
            let mut selected: HashSet<u64> = HashSet::new();
            for i in 0..k as usize {
                let mut j = self.randbelow(n)?;
                while selected.contains(&j) {
                    j = self.randbelow(n)?;
                }
                selected.insert(j);
                result[i] = j;
            }
        }
        Ok(result)
    }
}

// -- `_zipf_sample`, exact replica of schedule.py ----------------------------

fn zipf_sample(
    rng: &mut RsRandom,
    n: u64,
    k: u64,
    skew: f64,
    weights: &Option<Vec<f64>>,
) -> PyResult<Vec<usize>> {
    let weights = match weights {
        // The caller (Python `_routing`) always builds this table with Python's
        // own pow, guaranteeing identical f64 tables.
        Some(w) => w.clone(),
        None => (0..n as usize).map(|i| 1.0 / ((i as f64 + 1.0).powf(skew))).collect(),
    };
    if skew <= 0.0 {
        return Ok(rng.sample(n, k)?.into_iter().map(|x| x as usize).collect());
    }
    let mut chosen = Vec::with_capacity(k as usize);
    let mut idx: Vec<usize> = (0..n as usize).collect();
    for _ in 0..k {
        let mut total = 0.0f64;
        for &i in &idx {
            total += weights[i];
        }
        let roll = rng.random() * total;
        let mut acc = 0.0f64;
        let mut picked = *idx.last().unwrap();
        for &i in &idx {
            acc += weights[i];
            if roll <= acc {
                picked = i;
                break;
            }
        }
        chosen.push(picked);
        let pos = idx.iter().position(|&x| x == picked).unwrap();
        idx.remove(pos);
    }
    Ok(chosen)
}

/// `_routing(n_layers, n_experts, top_k, tokens, skew, seed, weights)`:
/// `routing[token][layer] = list of selected expert ids`.
#[pyfunction]
#[pyo3(signature = (seed, n_layers, n_experts, top_k, tokens, skew, weights = None))]
pub fn routing(
    seed: i64,
    n_layers: u64,
    n_experts: u64,
    top_k: u64,
    tokens: u64,
    skew: f64,
    weights: Option<Vec<f64>>,
) -> PyResult<Py<PyAny>> {
    let mut rng = RsRandom { mt: Mt::init_by_array(&seed_words(seed)) };
    Python::with_gil(|py| {
        let outer = PyList::empty(py);
        for _ in 0..tokens {
            let per_token = PyList::empty(py);
            for _ in 0..n_layers {
                let picks = zipf_sample(&mut rng, n_experts, top_k, skew, &weights)?;
                let pl = PyList::empty(py);
                for p in picks {
                    pl.append(p as u64)?;
                }
                per_token.append(pl)?;
            }
            outer.append(per_token)?;
        }
        Ok(outer.into_pyobject(py)?.unbind().into())
    })
}

// -- ExpertCache, insertion-ordered to match Python dict semantics -----------

type CacheKey = (i64, i64);

struct Cache {
    capacity: i64,
    policy: String,
    alloc: String,
    track_order: bool,
    pin_fraction: f64,
    aging_interval: i64,
    n_layers: Option<i64>,
    alloc_interval: i64,
    keys: IndexMap<CacheKey, ()>,
    order: Vec<CacheKey>,
    freq: IndexMap<CacheKey, i64>,
    layer_freq: IndexMap<i64, IndexMap<i64, i64>>,
    count: IndexMap<i64, i64>,
    cap: IndexMap<i64, i64>,
    pins: HashSet<CacheKey>,
    accesses: i64,
    hits: i64,
    misses: i64,
    bypasses: i64,
    reallocations: i64,
}

impl Cache {
    fn touch(&mut self, key: CacheKey) {
        if let Some(pos) = self.order.iter().position(|&k| k == key) {
            self.order.remove(pos);
        }
        self.order.push(key);
    }

    fn insert(&mut self, key: CacheKey) {
        self.keys.insert(key, ());
        if self.track_order {
            self.order.push(key);
        }
        *self.count.entry(key.0).or_insert(0) += 1;
    }

    fn remove(&mut self, key: CacheKey) {
        self.keys.shift_remove(&key);
        if self.track_order {
            if let Some(pos) = self.order.iter().position(|&k| k == key) {
                self.order.remove(pos);
            }
        }
        self.pins.remove(&key);
        *self.count.entry(key.0).or_insert(0) -= 1;
    }

    fn default_cap(&self) -> i64 {
        let layers = std::cmp::max(1, self.n_layers.clone().unwrap_or(0));
        std::cmp::max(1, self.capacity / layers)
    }

    /// Exact port of `ExpertCache._victim`, tie rule included: `min(...)`
    /// keeps the *first* minimal element in dict-insertion order.
    fn victim(&self, layer: Option<i64>) -> Option<CacheKey> {
        let per_layer = self.alloc == "per-layer" && layer.is_some();
        if self.policy == "lru" {
            for &k in &self.order {
                if !per_layer || k.0 == layer.unwrap() {
                    return Some(k);
                }
            }
            return None;
        }
        let pool: Vec<CacheKey> = if per_layer {
            self.keys.keys().filter(|k| k.0 == layer.unwrap()).copied().collect()
        } else {
            self.keys.keys().copied().collect()
        };
        if pool.is_empty() {
            return None;
        }
        let mut candidates: Vec<CacheKey> =
            pool.iter().copied().filter(|k| !self.pins.contains(k)).collect();
        if candidates.is_empty() {
            candidates = pool.clone();
        }
        let mut best = candidates[0];
        let mut best_freq = self.freq.get(&candidates[0]).copied().unwrap_or(0);
        for &k in &candidates[1..] {
            let f = self.freq.get(&k).copied().unwrap_or(0);
            if f < best_freq {
                best = k;
                best_freq = f;
            }
        }
        Some(best)
    }

    fn age(&mut self) {
        for v in self.freq.values_mut() {
            *v /= 2;
        }
        for lf in self.layer_freq.values_mut() {
            for v in lf.values_mut() {
                *v /= 2;
            }
        }
        let keep = std::cmp::max(1, (self.capacity as f64 * self.pin_fraction) as i64);
        // sorted(self._keys, key=freq, reverse=True) is stable: ties keep the
        // dict's insertion order. Rust's `sort_by` is stable too.
        let mut ks: Vec<CacheKey> = self.keys.keys().copied().collect();
        ks.sort_by(|a, b| {
            let fa = self.freq.get(a).copied().unwrap_or(0);
            let fb = self.freq.get(b).copied().unwrap_or(0);
            fb.cmp(&fa)
        });
        self.pins = ks.into_iter().take(keep as usize).collect();
    }

    fn reallocate(&mut self) {
        let mut ranked: Vec<(i64, Vec<i64>)> = Vec::new();
        for (l, f) in &self.layer_freq {
            if !f.is_empty() {
                let mut v: Vec<i64> = f.values().copied().collect();
                // Python sorted(values, reverse=True): stable, ties keep the
                // per-layer dict insertion order.
                v.sort_by(|a, b| b.cmp(a));
                ranked.push((*l, v));
            }
        }
        let total: i64 = ranked.iter().map(|(_, r)| r.iter().sum::<i64>()).sum();
        if ranked.is_empty() || total <= 0 || self.capacity <= 0 {
            return;
        }
        let mut assigned: IndexMap<i64, i64> =
            ranked.iter().map(|(l, _)| (*l, 0)).collect();
        for _ in 0..self.capacity {
            let mut best: Option<i64> = None;
            let mut best_mass = -1.0f64;
            for (l, r) in &ranked {
                let kk = assigned[l] as usize;
                let mass = if kk < r.len() { r[kk] as f64 / total as f64 } else { 0.0 };
                // strict `>` keeps the first layer on ties, like dict order.
                if mass > best_mass {
                    best_mass = mass;
                    best = Some(*l);
                }
            }
            let best = best.unwrap();
            *assigned.get_mut(&best).unwrap() += 1;
        }
        self.reallocations += 1;
        // Trim any layer now over its new quota.
        for (l, a) in &assigned {
            while self.count.get(l).copied().unwrap_or(0) > *a {
                let v = match self.victim(Some(*l)) {
                    Some(v) => v,
                    None => break,
                };
                self.remove(v);
            }
        }
    }

    fn admit(&mut self, key: CacheKey) {
        if self.alloc == "per-layer" {
            let layer = key.0;
            let default_cap = self.default_cap();
            self.cap.entry(layer).or_insert(default_cap);
            if *self.count.get(&layer).unwrap_or(&0) < self.cap[&layer] {
                self.insert(key);
                return;
            }
        } else if (self.keys.len() as i64) < self.capacity {
            self.insert(key);
            return;
        }
        let per_layer = self.alloc == "per-layer";
        let victim = if per_layer { self.victim(Some(key.0)) } else { self.victim(None) };
        let Some(victim) = victim else {
            self.insert(key);
            return;
        };
        if self.policy == "hot"
            && *self.freq.get(&key).unwrap_or(&0) <= *self.freq.get(&victim).unwrap_or(&0)
        {
            self.bypasses += 1;
            return;
        }
        self.remove(victim);
        self.insert(key);
    }

    fn maybe(&mut self) {
        if self.policy == "hot" && self.accesses % self.aging_interval == 0 {
            self.age();
        }
        if self.alloc == "per-layer" && self.accesses % self.alloc_interval == 0 {
            self.reallocate();
        }
    }
}

#[pyclass]
pub struct ExpertCacheRs {
    c: Cache,
}

#[pymethods]
impl ExpertCacheRs {
    #[new]
    #[pyo3(signature = (capacity, policy = "lru".to_string(), *,
        pin_fraction = 0.5, aging_interval = None, n_layers = None,
        alloc = "global".to_string(), alloc_interval = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        capacity: i64,
        policy: String,
        pin_fraction: f64,
        aging_interval: Option<i64>,
        n_layers: Option<i64>,
        alloc: String,
        alloc_interval: Option<i64>,
    ) -> PyResult<Self> {
        if policy != "lru" && policy != "lfu" && policy != "hot" {
            return Err(PyValueError::new_err(format!("unknown cache_policy {policy:?}")));
        }
        if alloc != "global" && alloc != "per-layer" {
            return Err(PyValueError::new_err(format!("unknown cache_alloc {alloc:?}")));
        }
        if alloc == "per-layer" && n_layers.map(|n| n == 0).unwrap_or(true) {
            return Err(PyValueError::new_err("per-layer allocation needs n_layers"));
        }
        let capacity = std::cmp::max(0, capacity);
        Ok(ExpertCacheRs {
            c: Cache {
                capacity,
                track_order: policy == "lru",
                policy,
                pin_fraction,
                aging_interval: aging_interval.unwrap_or_else(|| std::cmp::max(1, 8 * capacity)),
                n_layers,
                alloc_interval: alloc_interval.unwrap_or_else(|| std::cmp::max(1, 4 * capacity)),
                alloc,
                keys: IndexMap::new(),
                order: Vec::new(),
                freq: IndexMap::new(),
                layer_freq: IndexMap::new(),
                count: IndexMap::new(),
                cap: IndexMap::new(),
                pins: HashSet::new(),
                accesses: 0,
                hits: 0,
                misses: 0,
                bypasses: 0,
                reallocations: 0,
            },
        })
    }

    /// Record one reference; return True on a cache hit. Exact port of
    /// `ExpertCache.access`.]
    fn access(&mut self, key: (i64, i64)) -> PyResult<bool> {
        self.c.accesses += 1;
        *self.c.freq.entry(key).or_insert(0) += 1;
        let lf = self.c.layer_freq.entry(key.0).or_default();
        *lf.entry(key.1).or_insert(0) += 1;
        if self.c.keys.contains_key(&key) {
            self.c.hits += 1;
            if self.c.track_order {
                self.c.touch(key);
            }
            self.c.maybe();
            return Ok(true);
        }
        self.c.misses += 1;
        if self.c.capacity > 0 {
            self.c.admit(key);
        }
        self.c.maybe();
        Ok(false)
    }

    fn __contains__(&self, key: (i64, i64)) -> bool {
        self.c.keys.contains_key(&key)
    }

    fn __len__(&self) -> usize {
        self.c.keys.len()
    }

    // -- attributes the manifest and the tests read -------------------------

    #[getter]
    fn hits(&self) -> i64 {
        self.c.hits
    }

    #[getter]
    fn misses(&self) -> i64 {
        self.c.misses
    }

    #[getter]
    fn bypasses(&self) -> i64 {
        self.c.bypasses
    }

    #[getter]
    fn reallocations(&self) -> i64 {
        self.c.reallocations
    }

    #[getter]
    fn accesses(&self) -> i64 {
        self.c.accesses
    }

    #[getter]
    fn capacity(&self) -> i64 {
        self.c.capacity
    }

    // Snapshots for the internal-name tests; conversion cost only matters to
    // tests, the hot loop never reads these.
    #[getter]
    fn _order(&self, py: Python) -> PyResult<Py<PyList>> {
        let list = PyList::empty(py);
        for &k in &self.c.order {
            let kt = PyTuple::new(py, [k.0, k.1])?;
            list.append(kt)?;
        }
        Ok(list.unbind())
    }

    #[getter]
    fn _freq(&self, py: Python) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        for (&k, &v) in &self.c.freq {
            let kt = PyTuple::new(py, [k.0, k.1])?;
            dict.set_item(kt, v)?;
        }
        Ok(dict.unbind())
    }

    #[getter]
    fn _count(&self, py: Python) -> PyResult<Py<PyDict>> {
        build_int_dict(py, &self.c.count)
    }

    #[getter]
    fn _cap(&self, py: Python) -> PyResult<Py<PyDict>> {
        build_int_dict(py, &self.c.cap)
    }
}

fn build_int_dict(
    py: Python,
    src: &IndexMap<i64, i64>,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    for (&k, &v) in src {
        dict.set_item(k, v)?;
    }
    Ok(dict.unbind())
}

// -- module ------------------------------------------------------------------

#[pymodule]
fn _planner_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RsRandom>()?;
    m.add_class::<ExpertCacheRs>()?;
    m.add_function(wrap_pyfunction!(routing, m)?)?;
    Ok(())
}
