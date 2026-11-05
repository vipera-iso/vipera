//! Island for the one loop in the GGUF reader that cannot be skipped.
//!
//! A GGUF string array stores `count` entries as `[u64 len][len bytes]`, so its
//! end is only known by walking every length. The vocabularies of a real model
//! are 150k-400k entries of that, and the analyzer never reads a single one of
//! them -- it only needs the cursor to land after them.
//!
//! Python pays ~25 ms for that walk; this does the arithmetic and nothing else.
//! It takes the mmap directly (no copy, buffer protocol) and returns the end
//! offset. Decoding, bounds checking against the tensor table, and every other
//! GGUF rule stay in `vipera.gguf_fast` -- this function is deliberately unable
//! to change what the analyzer reads.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const OVERRUN: &str = "GGUF string array runs past the end of the file";

/// End offset of an array of GGUF strings that starts at `start`.
///
/// `count` is the array length from the file, `wide` selects the 64-bit length
/// prefix used by GGUF v2+ (v1 uses 32-bit).
#[pyfunction]
fn scan_strings(
    py: Python<'_>,
    buf: &Bound<'_, PyAny>,
    start: u64,
    count: u64,
    wide: bool,
) -> PyResult<u64> {
    let view = PyBuffer::<u8>::get(buf)?;
    let len = view.len_bytes();
    let data = view
        .as_slice(py)
        .ok_or_else(|| PyValueError::new_err("buffer is not contiguous"))?;
    let step = if wide { 8usize } else { 4usize };
    let base = data.as_ptr().cast::<u8>();
    let mut pos = usize::try_from(start).map_err(|_| PyValueError::new_err(OVERRUN))?;

    for _ in 0..count {
        if pos.checked_add(step).is_none_or(|end| end > len) {
            return Err(PyValueError::new_err(OVERRUN));
        }
        // SAFETY: `pos + step <= len`, and `base` points at `len` readable bytes
        // of the buffer for as long as the `PyBuffer` is alive. `read_unaligned`
        // because a GGUF length prefix sits at any byte offset.
        let length = unsafe {
            if wide {
                u64::from_le((base.add(pos) as *const u64).read_unaligned())
            } else {
                u32::from_le((base.add(pos) as *const u32).read_unaligned()) as u64
            }
        };
        pos = pos
            .checked_add(step)
            .and_then(|p| p.checked_add(usize::try_from(length).ok()?))
            .ok_or_else(|| PyValueError::new_err(OVERRUN))?;
        if pos > len {
            return Err(PyValueError::new_err(OVERRUN));
        }
    }
    Ok(pos as u64)
}

#[pymodule]
fn _gguf_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(scan_strings, m)?)?;
    Ok(())
}
