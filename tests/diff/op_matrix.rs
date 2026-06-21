//! Per-op differential matrix — test-side dispatch layer.
//!
//! The case generator (kernel builders, host oracles, `cases()`, the
//! `OpCase`/`RawValues` types) lives in `quanta_ir::op_matrix_cases` so
//! it can be shared with the WGSL browser audit. This module keeps only
//! the parts that need a `quanta::Gpu`: per-lane dispatch and the
//! `RawOutput` wrappers.

use super::lane::Lane;
use super::output::{RawOutput, RawValues};

pub use quanta_ir::op_matrix_cases::{OpCase, cases};

/// Wrap a dispatched buffer as a lane `RawOutput`. (Was `OpCase::output`
/// before the generator moved to quanta-ir; kept as a free function so it
/// can reference the test-side `Lane`/`RawOutput`.)
pub fn output(case: &OpCase, lane: Lane, values: RawValues) -> RawOutput {
    RawOutput {
        lane,
        kernel: Box::leak(case.name.clone().into_boxed_str()),
        values,
    }
}

/// The CPU-computed expected output as a Reference `RawOutput`.
pub fn oracle(case: &OpCase) -> RawOutput {
    RawOutput {
        lane: Lane::Reference,
        kernel: Box::leak(case.name.clone().into_boxed_str()),
        values: case.expected.clone(),
    }
}

// ── Per-lane dispatcher ──────────────────────────────────────────────

/// Dispatch one case on the given Gpu, return the raw output buffer.
///
/// Bind layout matches `build_binop_def`: slot 0 = a, slot 1 = b,
/// slot 2 = out. All three are length-1 typed fields.
#[cfg(any(feature = "software", feature = "metal", feature = "vulkan"))]
pub fn dispatch_on(gpu: &quanta::Gpu, case: &OpCase, lane: Lane) -> RawOutput {
    let bytes = quanta_ir::serialize_kernel(&case.def);
    let mut wave = gpu.wave_jit(&bytes).expect("wave_jit");

    // The dispatcher picks `Field<T>` allocations from the input
    // RawValues variants, and the output `Field<U>` from the
    // expected variant. Cmp produces U32 from any input type
    // (Bool→U32 cast inside the kernel); Cast produces target-type
    // from source-type.
    let values = dispatch_pair_typed(gpu, &mut wave, &case.input_a, &case.input_b, &case.expected);

    output(case, lane, values)
}

/// Match on (input_a, input_b, expected) variant triples and pick
/// the right typed allocation for each field. The four scalar
/// widths × six RawValues variants × asymmetric in/out types gives
/// a 36-arm match in principle; this enumerates only the (in_pair,
/// out) combinations we actually use today and panics on the rest.
#[cfg(any(feature = "software", feature = "metal", feature = "vulkan"))]
fn dispatch_pair_typed(
    gpu: &quanta::Gpu,
    wave: &mut quanta::Wave,
    in_a: &RawValues,
    in_b: &RawValues,
    expected: &RawValues,
) -> RawValues {
    match (in_a, in_b, expected) {
        // Symmetric: input and output share the type (BinOp, UnaryOp).
        (RawValues::U32(a), RawValues::U32(b), RawValues::U32(_)) => {
            dispatch_pair::<u32, u32>(gpu, wave, a, b, RawValues::U32)
        }
        (RawValues::U64(a), RawValues::U64(b), RawValues::U64(_)) => {
            dispatch_pair::<u64, u64>(gpu, wave, a, b, RawValues::U64)
        }
        (RawValues::I32(a), RawValues::I32(b), RawValues::I32(_)) => {
            dispatch_pair::<i32, i32>(gpu, wave, a, b, RawValues::I32)
        }
        (RawValues::I64(a), RawValues::I64(b), RawValues::I64(_)) => {
            dispatch_pair::<i64, i64>(gpu, wave, a, b, RawValues::I64)
        }
        (RawValues::F32(a), RawValues::F32(b), RawValues::F32(_)) => {
            dispatch_pair::<f32, f32>(gpu, wave, a, b, RawValues::F32)
        }
        (RawValues::F64(a), RawValues::F64(b), RawValues::F64(_)) => {
            dispatch_pair::<f64, f64>(gpu, wave, a, b, RawValues::F64)
        }
        // Cmp with non-U32 inputs: produces a U32 (0/1) output via
        // Cast(Bool→U32) in the kernel body. The U32-input variant
        // is handled by the symmetric arm above.
        (RawValues::I32(a), RawValues::I32(b), RawValues::U32(_)) => {
            dispatch_pair::<i32, u32>(gpu, wave, a, b, RawValues::U32)
        }
        (RawValues::F32(a), RawValues::F32(b), RawValues::U32(_)) => {
            dispatch_pair::<f32, u32>(gpu, wave, a, b, RawValues::U32)
        }
        // Cast across types: a∈From, b unused, out∈To. The dispatcher
        // still allocates a `Field<From>` for b because the kernel
        // emits a `Load` from slot 1 even though the result is dead.
        (RawValues::U32(a), RawValues::U32(b), RawValues::I32(_)) => {
            dispatch_pair::<u32, i32>(gpu, wave, a, b, RawValues::I32)
        }
        (RawValues::U32(a), RawValues::U32(b), RawValues::F32(_)) => {
            dispatch_pair::<u32, f32>(gpu, wave, a, b, RawValues::F32)
        }
        (RawValues::F32(a), RawValues::F32(b), RawValues::I32(_)) => {
            dispatch_pair::<f32, i32>(gpu, wave, a, b, RawValues::I32)
        }
        (RawValues::I32(a), RawValues::I32(b), RawValues::F32(_)) => {
            dispatch_pair::<i32, f32>(gpu, wave, a, b, RawValues::F32)
        }
        // bf16: storage is the portable u32-slot (one bf16 per 32-bit
        // word), so the host fields are `Field<u32>` carrying the bf16
        // bits zero-extended; readback narrows back to u16.
        (RawValues::BF16(a), RawValues::BF16(b), RawValues::BF16(_)) => {
            dispatch_bf16(gpu, wave, a, b)
        }
        // fp8: same portable u32-slot storage as bf16, one byte per word.
        (RawValues::FP8E5M2(a), RawValues::FP8E5M2(b), RawValues::FP8E5M2(_)) => {
            dispatch_fp8(gpu, wave, a, b, RawValues::FP8E5M2)
        }
        (RawValues::FP8E4M3(a), RawValues::FP8E4M3(b), RawValues::FP8E4M3(_)) => {
            dispatch_fp8(gpu, wave, a, b, RawValues::FP8E4M3)
        }
        // Quantize: f32 in → integer code out. The int code field uses the
        // i32 storage slot (Q8) or u32 packed-nibble slot (Q4).
        (RawValues::F32(a), RawValues::F32(b), RawValues::Q8(_)) => {
            dispatch_quantize(gpu, wave, a, b, false)
        }
        (RawValues::F32(a), RawValues::F32(b), RawValues::Q4(_)) => {
            dispatch_quantize(gpu, wave, a, b, true)
        }
        // Dequantize: integer code in → f32 out.
        (RawValues::Q8(a), RawValues::Q8(b), RawValues::F32(_)) => {
            dispatch_dequantize(gpu, wave, a, b, false)
        }
        (RawValues::Q4(a), RawValues::Q4(b), RawValues::F32(_)) => {
            dispatch_dequantize(gpu, wave, a, b, true)
        }
        _ => panic!(
            "op_matrix::read_output: in/out type combo not yet wired \
             (a={}, b={}, out={})",
            in_a.type_tag(),
            in_b.type_tag(),
            expected.type_tag()
        ),
    }
}

/// Allocate `Field<TIn>` × 2 + `Field<TOut>` × 1, upload, bind,
/// dispatch one quark, read back as `Vec<TOut>`. Caller picks the
/// `RawValues` wrapper for the output variant.
#[cfg(any(feature = "software", feature = "metal", feature = "vulkan"))]
fn dispatch_pair<TIn: Copy + 'static, TOut: Copy + 'static>(
    gpu: &quanta::Gpu,
    wave: &mut quanta::Wave,
    a: &[TIn],
    b: &[TIn],
    wrap: fn(Vec<TOut>) -> RawValues,
) -> RawValues {
    let fa = gpu.field::<TIn>(1).unwrap();
    let fb = gpu.field::<TIn>(1).unwrap();
    let fout = gpu.field::<TOut>(1).unwrap();
    fa.write(a).unwrap();
    fb.write(b).unwrap();
    wave.bind(0, &fa);
    wave.bind(1, &fb);
    wave.bind(2, &fout);
    let mut pulse = gpu.dispatch(wave, 1).unwrap();
    pulse.wait().unwrap();
    wrap(fout.read().unwrap())
}

/// bf16 dispatch over the portable u32-slot storage: each bf16 value is a
/// `u32` carrying its bits in the low 16. Uploads bf16-bits-as-u32, reads
/// back, and narrows to `u16` for the `RawValues::BF16` result.
#[cfg(any(feature = "software", feature = "metal", feature = "vulkan"))]
fn dispatch_bf16(gpu: &quanta::Gpu, wave: &mut quanta::Wave, a: &[u16], b: &[u16]) -> RawValues {
    let a32: Vec<u32> = a.iter().map(|&x| x as u32).collect();
    let b32: Vec<u32> = b.iter().map(|&x| x as u32).collect();
    let fa = gpu.field::<u32>(1).unwrap();
    let fb = gpu.field::<u32>(1).unwrap();
    let fout = gpu.field::<u32>(1).unwrap();
    fa.write(&a32).unwrap();
    fb.write(&b32).unwrap();
    wave.bind(0, &fa);
    wave.bind(1, &fb);
    wave.bind(2, &fout);
    let mut pulse = gpu.dispatch(wave, 1).unwrap();
    pulse.wait().unwrap();
    let out: Vec<u16> = fout.read().unwrap().into_iter().map(|w| w as u16).collect();
    RawValues::BF16(out)
}

/// fp8 dispatch over the portable u32-slot storage: each fp8 byte is a
/// `u32` carrying its bits in the low 8. Uploads fp8-bits-as-u32, reads
/// back, and narrows to `u8` for the `RawValues::FP8*` result.
#[cfg(any(feature = "software", feature = "metal", feature = "vulkan"))]
fn dispatch_fp8(
    gpu: &quanta::Gpu,
    wave: &mut quanta::Wave,
    a: &[u8],
    b: &[u8],
    wrap: fn(Vec<u8>) -> RawValues,
) -> RawValues {
    let a32: Vec<u32> = a.iter().map(|&x| x as u32).collect();
    let b32: Vec<u32> = b.iter().map(|&x| x as u32).collect();
    let fa = gpu.field::<u32>(1).unwrap();
    let fb = gpu.field::<u32>(1).unwrap();
    let fout = gpu.field::<u32>(1).unwrap();
    fa.write(&a32).unwrap();
    fb.write(&b32).unwrap();
    wave.bind(0, &fa);
    wave.bind(1, &fb);
    wave.bind(2, &fout);
    let mut pulse = gpu.dispatch(wave, 1).unwrap();
    pulse.wait().unwrap();
    let out: Vec<u8> = fout.read().unwrap().into_iter().map(|w| w as u8).collect();
    wrap(out)
}

/// Quantize dispatch: f32 inputs in, integer code out. Q8 codes ride the
/// i32 storage slot; Q4 codes ride a u32 packed-nibble slot (nibble 0).
/// Reads the code back and narrows to i8 for `RawValues::Q8`/`Q4`.
#[cfg(any(feature = "software", feature = "metal", feature = "vulkan"))]
fn dispatch_quantize(
    gpu: &quanta::Gpu,
    wave: &mut quanta::Wave,
    a: &[f32],
    b: &[f32],
    q4: bool,
) -> RawValues {
    let fa = gpu.field::<f32>(1).unwrap();
    let fb = gpu.field::<f32>(1).unwrap();
    fa.write(a).unwrap();
    fb.write(b).unwrap();
    wave.bind(0, &fa);
    wave.bind(1, &fb);
    if q4 {
        let fout = gpu.field::<u32>(1).unwrap();
        wave.bind(2, &fout);
        gpu.dispatch(wave, 1).unwrap().wait().unwrap();
        // nibble 0 of the word, sign-extended to i8.
        let out: Vec<i8> = fout
            .read()
            .unwrap()
            .into_iter()
            .map(|w| (((w & 0xF) ^ 0x8).wrapping_sub(0x8)) as i8)
            .collect();
        RawValues::Q4(out)
    } else {
        let fout = gpu.field::<i32>(1).unwrap();
        wave.bind(2, &fout);
        gpu.dispatch(wave, 1).unwrap().wait().unwrap();
        let out: Vec<i8> = fout.read().unwrap().into_iter().map(|w| w as i8).collect();
        RawValues::Q8(out)
    }
}

/// Dequantize dispatch: integer code in, f32 out. Codes are uploaded into
/// the i32 slot (Q8) or u32 packed-nibble slot (Q4).
#[cfg(any(feature = "software", feature = "metal", feature = "vulkan"))]
fn dispatch_dequantize(
    gpu: &quanta::Gpu,
    wave: &mut quanta::Wave,
    a: &[i8],
    b: &[i8],
    q4: bool,
) -> RawValues {
    let fout = gpu.field::<f32>(1).unwrap();
    // Bind through the input fields; these must outlive the dispatch, so
    // hold them in this scope rather than an inner block.
    if q4 {
        // pack the code into nibble 0 of a u32 word.
        let a32: Vec<u32> = a.iter().map(|&x| (x as u32) & 0xF).collect();
        let b32: Vec<u32> = b.iter().map(|&x| (x as u32) & 0xF).collect();
        let fa = gpu.field::<u32>(1).unwrap();
        let fb = gpu.field::<u32>(1).unwrap();
        fa.write(&a32).unwrap();
        fb.write(&b32).unwrap();
        wave.bind(0, &fa);
        wave.bind(1, &fb);
        wave.bind(2, &fout);
        gpu.dispatch(wave, 1).unwrap().wait().unwrap();
    } else {
        let a32: Vec<i32> = a.iter().map(|&x| x as i32).collect();
        let b32: Vec<i32> = b.iter().map(|&x| x as i32).collect();
        let fa = gpu.field::<i32>(1).unwrap();
        let fb = gpu.field::<i32>(1).unwrap();
        fa.write(&a32).unwrap();
        fb.write(&b32).unwrap();
        wave.bind(0, &fa);
        wave.bind(1, &fb);
        wave.bind(2, &fout);
        gpu.dispatch(wave, 1).unwrap().wait().unwrap();
    }
    RawValues::F32(fout.read().unwrap())
}
