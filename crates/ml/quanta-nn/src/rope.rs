//! Fused rotary position embeddings (RoPE, rotate-half convention).
//!
//! One elementwise kernel serves BOTH directions: because each frequency
//! pair `(j, j + d/2)` undergoes an orthogonal rotation, the VJP is the
//! rotation by `−θ` — proven in `specs/verify/lean/Quanta/Nn/
//! RotationVjp.lean` (T9216 adjoint, T9218 inverse-composition), so the
//! backward is the same kernel with `sign = −1`. T9217 (norm preservation)
//! is the stability story: RoPE is an isometry per pair and can amplify
//! neither activations nor gradients.
//!
//! Semantics match the composed [`quanta_array::autograd::Var::rope`] (the
//! differential-test oracle): `y = x⊙cos + rotate_half(x)⊙sin` with
//! `rotate_half` pairing `j` and `j + d/2`. This fused core handles the 2-D
//! `[T, d]` case; leading batch/head dims are a host loop, like the SDPA
//! and norm cores.

use crate::functional::{adopt_f32_field, f32_input, lift};
use quanta_array::autograd::{AutogradError, DiffScalar, RopeCache, Tape, Var};
use quanta_array::{Array, ToF64};
use quanta_core::{Field, Gpu, QuantaError};

#[allow(unused_imports)]
mod dsl {
    use quanta_core::*;

    /// Rotate every frequency pair of `x` by its cached angle. `sign = 1.0`
    /// is the forward rotation; `sign = -1.0` the adjoint/inverse (T9218).
    /// One thread per element; the pair partner sits `d/2` away; loop-free.
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    pub fn rope_apply(
        x: &[f32],
        cosv: &[f32],
        sinv: &[f32],
        out: &mut [f32],
        n: u32,
        d: u32,
        sign: f32,
    ) {
        let i = quark_id();
        let total = n * d;
        let idx = if i < total { i } else { 0u32 };
        let col = idx % d;
        let half = d / 2u32;

        // First-half lanes pair forward (+half) with a minus; second-half
        // lanes pair backward (−half) with a plus — the rotate-half signs.
        let first = if col < half { 1u32 } else { 0u32 };
        let partner = if col < half { idx + half } else { idx - half };
        let pm = if first > 0u32 { -1.0f32 } else { 1.0f32 };

        let y = x[idx as usize] * cosv[idx as usize]
            + sign * pm * x[partner as usize] * sinv[idx as usize];
        if i < total {
            out[idx as usize] = y;
        }
    }
}

/// Host dispatch: apply the (signed) rotation. `x`/`out` are `n×d`
/// row-major; `cosv`/`sinv` are the cache rows for these `n` positions
/// (`n×d`, rotate-half layout: entries `j` and `j + d/2` share a
/// frequency). `sign = +1` forward, `−1` adjoint.
#[allow(clippy::too_many_arguments)]
pub fn rope_apply(
    gpu: &Gpu,
    n: u32,
    d: u32,
    sign: f32,
    x: &Field<f32>,
    cosv: &Field<f32>,
    sinv: &Field<f32>,
    out: &Field<f32>,
) -> Result<(), QuantaError> {
    let (nu, du) = (n as usize, d as usize);
    if d == 0 || !du.is_multiple_of(2) {
        return Err(QuantaError::invalid_param("rope: head dim must be even"));
    }
    if x.len() != nu * du || out.len() != nu * du {
        return Err(QuantaError::invalid_param("rope: X/OUT length must be n*d"));
    }
    if cosv.len() != nu * du || sinv.len() != nu * du {
        return Err(QuantaError::invalid_param(
            "rope: COS/SIN length must be n*d",
        ));
    }
    if n == 0 {
        return Ok(());
    }
    let mut w = dsl::rope_apply(gpu)?;
    w.bind(0, x);
    w.bind(1, cosv);
    w.bind(2, sinv);
    w.bind(3, out);
    w.set_value(4, n);
    w.set_value(5, d);
    w.set_value(6, sign);
    gpu.dispatch(&w, n * d)?;
    Ok(())
}

fn bad(msg: &'static str) -> AutogradError {
    AutogradError::from(quanta_array::ArrayError::Gpu(QuantaError::invalid_param(
        msg,
    )))
}

/// Tape-differentiable fused RoPE over a `[T, d]` input, using the
/// positions `0..T` of `cache`. The backward reuses the same kernel with
/// `sign = −1` (T9216/T9218); the composed [`Var::rope`] is the oracle.
pub fn rope_var<T: DiffScalar + ToF64>(
    tape: &Tape<T>,
    x: &Var<T>,
    cache: &RopeCache<T>,
) -> Result<Var<T>, AutogradError> {
    let xs = x.value().shape().to_vec();
    if xs.len() != 2 {
        return Err(bad(
            "rope_var: input must be 2-D [T, d] (batch = host loop)",
        ));
    }
    let (n, d) = (xs[0], xs[1]);
    if d != cache.d {
        return Err(bad("rope_var: last dim must equal the cache head dim"));
    }
    if n > cache.t {
        return Err(bad("rope_var: sequence length exceeds the cache"));
    }
    let gpu = x.value().gpu().clone();

    // Zero-copy bindings; the narrowed cos/sin windows contiguous-ify
    // on device and are captured by the backward (the VJP is the
    // rotation by −θ over the same tables).
    let xi = f32_input(&gpu, &x.value())?;
    let ci = f32_input(
        &gpu,
        &cache.cos.narrow(0, 0, n).map_err(AutogradError::from)?,
    )?;
    let si = f32_input(
        &gpu,
        &cache.sin.narrow(0, 0, n).map_err(AutogradError::from)?,
    )?;

    let of = gpu.field::<f32>(n * d).map_err(lift)?;
    rope_apply(
        &gpu,
        n as u32,
        d as u32,
        1.0,
        xi.field(),
        ci.field(),
        si.field(),
        &of,
    )
    .map_err(lift)?;
    let out_arr = adopt_f32_field::<T>(&gpu, of, &[n, d])?;

    let gpu_b = gpu.clone();
    let backward = move |g: &Array<T>| -> Result<Vec<Array<T>>, AutogradError> {
        let ginp = f32_input(&gpu_b, g)?;
        let dxf = gpu_b.field::<f32>(n * d).map_err(lift)?;
        rope_apply(
            &gpu_b,
            n as u32,
            d as u32,
            -1.0,
            ginp.field(),
            ci.field(),
            si.field(),
            &dxf,
        )
        .map_err(lift)?;
        let dx = adopt_f32_field::<T>(&gpu_b, dxf, &[n, d])?;
        Ok(vec![dx])
    };

    Ok(tape.custom_vjp(&[x], out_arr, backward))
}
