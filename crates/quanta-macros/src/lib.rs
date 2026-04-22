//! Proc macros for Quanta GPU kernels.

extern crate proc_macro;

#[allow(dead_code)]
mod compiler;
mod gpu_type;
mod parse;
mod validate;

use proc_macro::TokenStream;
use quote::{ToTokens, quote};
use syn::{Expr, ItemFn, Lit, parse::Parser, parse_macro_input};

/// Mark a function as a GPU kernel.
///
/// ```ignore
/// #[quanta::kernel]                                      // default: O3, workgroup [64,1,1]
/// #[quanta::kernel(opt = "O2")]                          // explicit O2
/// #[quanta::kernel(opt = "O0")]                          // no optimization (debug)
/// #[quanta::kernel(workgroup = [256])]                   // [256, 1, 1]
/// #[quanta::kernel(workgroup = [16, 16])]                // [16, 16, 1]
/// #[quanta::kernel(workgroup = [16, 16, 1])]             // explicit 3D
/// #[quanta::kernel(workgroup = [256], opt = "O2")]       // both
/// #[quanta::kernel(jit)]                                 // JIT: compile at runtime
/// ```
#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);

    // Parse attributes: optimization level, workgroup size, and jit flag
    let attr_str = attr.to_string();
    let is_jit = attr_str.contains("jit");
    let kernel_attrs = parse_kernel_attrs(attr.clone());

    if let Err(err) = validate::validate_kernel(&func) {
        return err.to_compile_error().into();
    }

    let mut kernel_def = match parse::parse_kernel(&func) {
        Ok(def) => def,
        Err(err) => return err.to_compile_error().into(),
    };
    kernel_def.opt_level = kernel_attrs.opt_level;
    kernel_def.workgroup_size = kernel_attrs.workgroup_size;

    if is_jit {
        return emit_jit_kernel(&func, &kernel_def);
    }

    let outputs = match compiler::compile_kernel(&kernel_def) {
        Ok(outputs) => outputs,
        Err(err) => {
            let msg = format!("quanta compiler error: {}", err);
            return syn::Error::new_spanned(&func.sig.ident, msg)
                .to_compile_error()
                .into();
        }
    };

    let func_name = &func.sig.ident;
    let binary_name = syn::Ident::new(
        &format!("{}_BINARY", func_name.to_string().to_uppercase()),
        func_name.span(),
    );

    let nvidia_expr = match &outputs.nvidia {
        Some(bytes) => {
            let lit = proc_macro2::Literal::byte_string(bytes);
            quote! { Some(#lit as &[u8]) }
        }
        None => quote! { None },
    };
    let amd_expr = match &outputs.amd {
        Some(bytes) => {
            let lit = proc_macro2::Literal::byte_string(bytes);
            quote! { Some(#lit as &[u8]) }
        }
        None => quote! { None },
    };
    let spirv_expr = match &outputs.spirv {
        Some(bytes) => {
            let lit = proc_macro2::Literal::byte_string(bytes);
            quote! { Some(#lit as &[u8]) }
        }
        None => quote! { None },
    };
    let metallib_expr = match &outputs.metallib {
        Some(bytes) => {
            let lit = proc_macro2::Literal::byte_string(bytes);
            quote! { Some(#lit as &[u8]) }
        }
        None => quote! { None },
    };

    let wg_x = kernel_attrs.workgroup_size[0];
    let wg_y = kernel_attrs.workgroup_size[1];
    let wg_z = kernel_attrs.workgroup_size[2];

    let expanded = quote! {
        pub static #binary_name: ::quanta::KernelBinary = ::quanta::KernelBinary {
            amd: #amd_expr,
            nvidia: #nvidia_expr,
            spirv: #spirv_expr,
            metallib: #metallib_expr,
        };

        pub fn #func_name(device: &::quanta::Gpu) -> Result<::quanta::Wave, ::quanta::QuantaError> {
            let binary = #binary_name.for_vendor(device.caps().vendor)
                .ok_or_else(|| ::quanta::QuantaError::compilation_failed(
                    format!("no compiled kernel for vendor {:?}", device.caps().vendor)
                ))?;
            let mut wave = device.wave(binary)?;
            wave.workgroup_size = [#wg_x, #wg_y, #wg_z];
            Ok(wave)
        }
    };

    expanded.into()
}

/// Emit JIT kernel: serialize KernelDef and embed it, generate runtime
/// compilation function via `wave_jit`.
fn emit_jit_kernel(func: &ItemFn, kernel_def: &quanta_ir::KernelDef) -> TokenStream {
    let func_name = &func.sig.ident;
    let def_name = syn::Ident::new(
        &format!("{}_DEF", func_name.to_string().to_uppercase()),
        func_name.span(),
    );

    let serialized = quanta_ir::serialize_kernel(kernel_def);
    let def_lit = proc_macro2::Literal::byte_string(&serialized);

    let wg_x = kernel_def.workgroup_size[0];
    let wg_y = kernel_def.workgroup_size[1];
    let wg_z = kernel_def.workgroup_size[2];

    let expanded = quote! {
        pub static #def_name: &[u8] = #def_lit;

        pub fn #func_name(device: &::quanta::Gpu) -> Result<::quanta::Wave, ::quanta::QuantaError> {
            let mut wave = device.wave_jit(#def_name)?;
            wave.workgroup_size = [#wg_x, #wg_y, #wg_z];
            Ok(wave)
        }
    };

    expanded.into()
}

/// Mark a function as a GPU device function (callable from kernels).
///
/// ```ignore
/// #[quanta::device]
/// fn activate(x: f32, threshold: f32) -> f32 {
///     if x > threshold { x } else { x * 0.99 }
/// }
/// ```
///
/// Device functions are inlined into kernels by LLVM.
/// They cannot be launched from CPU — only called from `#[quanta::kernel]` functions.
///
/// The proc macro captures the function source and emits a hidden constant
/// `__QUANTA_DEVICE_{NAME_UPPERCASE}` containing the source text. Kernel
/// compilation picks this up so MSL/WGSL emitters can prepend it as a regular
/// helper function.
#[proc_macro_attribute]
pub fn device(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let source = func.to_token_stream().to_string();
    let fn_name = &func.sig.ident;
    let const_name = syn::Ident::new(
        &format!("__QUANTA_DEVICE_{}", fn_name.to_string().to_uppercase()),
        fn_name.span(),
    );

    let expanded = quote! {
        pub const #const_name: &str = #source;
    };
    expanded.into()
}

/// Mark a variable as shared (workgroup-local) memory inside a kernel.
///
/// ```ignore
/// #[quanta::kernel]
/// fn reduce(data: &[f32], result: &mut [f32]) {
///     #[quanta::shared] let local: [f32; 256];
///     let lid = local_id();
///     local[lid] = data[quark_id()];
///     barrier();
/// }
/// ```
///
/// When used inside a `#[quanta::kernel]` body, the kernel parser handles
/// this attribute directly — it emits `SharedDecl`, `SharedLoad`, and
/// `SharedStore` ops in the IR.
///
/// The proc macro itself is a no-op pass-through; the real work happens
/// in the kernel parser which inspects `let` attributes.
#[proc_macro_attribute]
pub fn shared(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

/// Mark a function as a vertex shader.
///
/// Compiles the function to MSL and WGSL at build time and embeds both as a
/// `ShaderBinary` static. Value parameters become vertex attributes;
/// reference parameters (`&T`) become uniform buffer bindings.
///
/// ```ignore
/// #[quanta::vertex]
/// fn transform(
///     pos: Vec3,
///     normal: Vec3,
///     mvp: &Mat4,
/// ) -> Vec4 {
///     mvp * Vec4::new(pos.x, pos.y, pos.z, 1.0)
/// }
/// ```
#[proc_macro_attribute]
pub fn vertex(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);

    if matches!(func.sig.output, syn::ReturnType::Default) {
        return syn::Error::new_spanned(
            &func.sig.ident,
            "vertex shader must have a return type (clip-space position)",
        )
        .to_compile_error()
        .into();
    }

    let func_name = &func.sig.ident;
    let func_name_str = func_name.to_string();
    let binary_name = syn::Ident::new(
        &format!("{}_SHADER", func_name_str.to_uppercase()),
        func_name.span(),
    );

    // Parse shader params and body, then compile via the compiler binary.
    let params = match compiler::parse_shader_params(&func) {
        Ok(p) => p,
        Err(e) => return e.to_compile_error().into(),
    };
    let return_ty = match compiler::parse_return_type(&func) {
        Ok(t) => t,
        Err(e) => return e.to_compile_error().into(),
    };

    // Extract body source text for the compiler.
    let body_source = func.block.to_token_stream().to_string();

    let (spirv_expr, metallib_expr) =
        match compiler::compile_shader(&func_name_str, "vertex", &params, &return_ty, &body_source)
        {
            Some(output) => {
                let spirv = match &output.spirv {
                    Some(bytes) => {
                        let lit = proc_macro2::Literal::byte_string(bytes);
                        quote! { Some(#lit as &[u8]) }
                    }
                    None => quote! { None },
                };
                let metallib = match &output.metallib {
                    Some(bytes) => {
                        let lit = proc_macro2::Literal::byte_string(bytes);
                        quote! { Some(#lit as &[u8]) }
                    }
                    None => quote! { None },
                };
                (spirv, metallib)
            }
            None => (quote! { None }, quote! { None }),
        };

    let expanded = quote! {
        pub static #binary_name: ::quanta::ShaderBinary = ::quanta::ShaderBinary {
            spirv: #spirv_expr,
            metallib: #metallib_expr,
            entry_point: #func_name_str,
            stage: ::quanta::ShaderStage::Vertex,
        };

        pub fn #func_name() -> &'static ::quanta::ShaderBinary {
            &#binary_name
        }
    };
    expanded.into()
}

/// Mark a function as a fragment shader.
///
/// Compiles the function to GPU binary at build time and embeds it as a
/// `ShaderBinary` static. Value parameters become fragment stage inputs
/// (interpolated varyings); reference parameters become uniform/texture bindings.
///
/// ```ignore
/// #[quanta::fragment]
/// fn shade(
///     uv: Vec2,
///     color: Vec4,
/// ) -> Vec4 {
///     color * Vec4::new(uv.x, uv.y, 0.0, 1.0)
/// }
/// ```
#[proc_macro_attribute]
pub fn fragment(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);

    if matches!(func.sig.output, syn::ReturnType::Default) {
        return syn::Error::new_spanned(
            &func.sig.ident,
            "fragment shader must have a return type (output color)",
        )
        .to_compile_error()
        .into();
    }

    let func_name = &func.sig.ident;
    let func_name_str = func_name.to_string();
    let binary_name = syn::Ident::new(
        &format!("{}_SHADER", func_name_str.to_uppercase()),
        func_name.span(),
    );

    // Parse shader params and body, then compile via the compiler binary.
    let params = match compiler::parse_shader_params(&func) {
        Ok(p) => p,
        Err(e) => return e.to_compile_error().into(),
    };
    let return_ty = match compiler::parse_return_type(&func) {
        Ok(t) => t,
        Err(e) => return e.to_compile_error().into(),
    };

    let body_source = func.block.to_token_stream().to_string();

    let (spirv_expr, metallib_expr) = match compiler::compile_shader(
        &func_name_str,
        "fragment",
        &params,
        &return_ty,
        &body_source,
    ) {
        Some(output) => {
            let spirv = match &output.spirv {
                Some(bytes) => {
                    let lit = proc_macro2::Literal::byte_string(bytes);
                    quote! { Some(#lit as &[u8]) }
                }
                None => quote! { None },
            };
            let metallib = match &output.metallib {
                Some(bytes) => {
                    let lit = proc_macro2::Literal::byte_string(bytes);
                    quote! { Some(#lit as &[u8]) }
                }
                None => quote! { None },
            };
            (spirv, metallib)
        }
        None => (quote! { None }, quote! { None }),
    };

    let expanded = quote! {
        pub static #binary_name: ::quanta::ShaderBinary = ::quanta::ShaderBinary {
            spirv: #spirv_expr,
            metallib: #metallib_expr,
            entry_point: #func_name_str,
            stage: ::quanta::ShaderStage::Fragment,
        };

        pub fn #func_name() -> &'static ::quanta::ShaderBinary {
            &#binary_name
        }
    };
    expanded.into()
}

// === Tessellation shader macros (M4.1) ===

/// Mark a function as a tessellation control (hull) shader.
///
/// The function defines per-control-point logic and sets tessellation factors.
/// Source is captured at build time for MSL/WGSL emission.
///
/// ```ignore
/// #[quanta::tess_control]
/// fn hull(patch_id: u32) -> TessFactors {
///     TessFactors { edge: [4.0; 4], inside: [4.0; 2] }
/// }
/// ```
#[proc_macro_attribute]
pub fn tess_control(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let func_name = &func.sig.ident;
    let func_name_str = func_name.to_string();
    let binary_name = syn::Ident::new(
        &format!("{}_SHADER", func_name_str.to_uppercase()),
        func_name.span(),
    );

    let expanded = quote! {
        pub static #binary_name: ::quanta::ShaderBinary = ::quanta::ShaderBinary {
            spirv: None,
            metallib: None,
            entry_point: #func_name_str,
            stage: ::quanta::ShaderStage::TessControl,
        };

        pub fn #func_name() -> &'static ::quanta::ShaderBinary {
            &#binary_name
        }
    };
    expanded.into()
}

/// Mark a function as a tessellation evaluation (domain) shader.
///
/// Runs once per generated vertex after tessellation. Reads patch data
/// and barycentric coordinates to compute final vertex positions.
///
/// ```ignore
/// #[quanta::tess_eval]
/// fn domain(uv: Vec2, patch: &[Vec3; 4]) -> Vec4 {
///     // Bilinear interpolation of control points
///     let p = mix(mix(patch[0], patch[1], uv.x), mix(patch[2], patch[3], uv.x), uv.y);
///     Vec4::new(p.x, p.y, p.z, 1.0)
/// }
/// ```
#[proc_macro_attribute]
pub fn tess_eval(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let func_name = &func.sig.ident;
    let func_name_str = func_name.to_string();
    let binary_name = syn::Ident::new(
        &format!("{}_SHADER", func_name_str.to_uppercase()),
        func_name.span(),
    );

    let expanded = quote! {
        pub static #binary_name: ::quanta::ShaderBinary = ::quanta::ShaderBinary {
            spirv: None,
            metallib: None,
            entry_point: #func_name_str,
            stage: ::quanta::ShaderStage::TessEval,
        };

        pub fn #func_name() -> &'static ::quanta::ShaderBinary {
            &#binary_name
        }
    };
    expanded.into()
}

// === Mesh shader macros (M4.2) ===

/// Mark a function as a task (amplification) shader.
///
/// The task shader performs coarse-grained culling and determines how many
/// mesh shader threadgroups to launch. Optional — mesh shaders can be
/// dispatched directly without a task shader.
///
/// ```ignore
/// #[quanta::task]
/// fn cull(group_id: u32, instances: &[BoundingBox]) {
///     if is_visible(instances[group_id]) {
///         emit_mesh_threadgroups(1);
///     }
/// }
/// ```
#[proc_macro_attribute]
pub fn task(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let func_name = &func.sig.ident;
    let func_name_str = func_name.to_string();
    let binary_name = syn::Ident::new(
        &format!("{}_SHADER", func_name_str.to_uppercase()),
        func_name.span(),
    );

    let expanded = quote! {
        pub static #binary_name: ::quanta::ShaderBinary = ::quanta::ShaderBinary {
            spirv: None,
            metallib: None,
            entry_point: #func_name_str,
            stage: ::quanta::ShaderStage::Task,
        };

        pub fn #func_name() -> &'static ::quanta::ShaderBinary {
            &#binary_name
        }
    };
    expanded.into()
}

/// Mark a function as a mesh shader.
///
/// Replaces the vertex + input assembly stages. The mesh shader generates
/// vertices and primitives directly, enabling GPU-driven geometry processing.
///
/// ```ignore
/// #[quanta::mesh]
/// fn generate(group_id: u32) {
///     // Emit vertices and triangle indices directly
///     set_vertex(0, Vec4::new(-1.0, -1.0, 0.0, 1.0));
///     set_vertex(1, Vec4::new( 1.0, -1.0, 0.0, 1.0));
///     set_vertex(2, Vec4::new( 0.0,  1.0, 0.0, 1.0));
///     set_primitive(0, [0, 1, 2]);
/// }
/// ```
#[proc_macro_attribute]
pub fn mesh(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let func_name = &func.sig.ident;
    let func_name_str = func_name.to_string();
    let binary_name = syn::Ident::new(
        &format!("{}_SHADER", func_name_str.to_uppercase()),
        func_name.span(),
    );

    let expanded = quote! {
        pub static #binary_name: ::quanta::ShaderBinary = ::quanta::ShaderBinary {
            spirv: None,
            metallib: None,
            entry_point: #func_name_str,
            stage: ::quanta::ShaderStage::Mesh,
        };

        pub fn #func_name() -> &'static ::quanta::ShaderBinary {
            &#binary_name
        }
    };
    expanded.into()
}

// === Ray tracing shader macros (M4.3) ===

/// Mark a function as a ray generation shader.
///
/// The entry point for ray tracing — launched once per pixel (or per ray).
/// Uses `trace_ray()` to fire rays into the acceleration structure.
///
/// ```ignore
/// #[quanta::ray_gen]
/// fn camera_rays(pixel: UVec2, scene: &AccelerationStructure) {
///     let ray = compute_ray(pixel);
///     trace_ray(scene, ray, 0, 1000.0);
/// }
/// ```
#[proc_macro_attribute]
pub fn ray_gen(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let func_name = &func.sig.ident;
    let func_name_str = func_name.to_string();
    let binary_name = syn::Ident::new(
        &format!("{}_SHADER", func_name_str.to_uppercase()),
        func_name.span(),
    );

    let expanded = quote! {
        pub static #binary_name: ::quanta::ShaderBinary = ::quanta::ShaderBinary {
            spirv: None,
            metallib: None,
            entry_point: #func_name_str,
            stage: ::quanta::ShaderStage::RayGen,
        };

        pub fn #func_name() -> &'static ::quanta::ShaderBinary {
            &#binary_name
        }
    };
    expanded.into()
}

/// Mark a function as a closest-hit shader.
///
/// Invoked when a ray intersects the nearest surface. Computes the shading
/// result (color, material response) and may fire secondary rays (reflections).
///
/// ```ignore
/// #[quanta::closest_hit]
/// fn shade(hit: HitInfo, ray: Ray) -> Vec4 {
///     let normal = hit.normal;
///     let color = sample_texture(hit.uv);
///     color * dot(normal, light_dir).max(0.0)
/// }
/// ```
#[proc_macro_attribute]
pub fn closest_hit(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let func_name = &func.sig.ident;
    let func_name_str = func_name.to_string();
    let binary_name = syn::Ident::new(
        &format!("{}_SHADER", func_name_str.to_uppercase()),
        func_name.span(),
    );

    let expanded = quote! {
        pub static #binary_name: ::quanta::ShaderBinary = ::quanta::ShaderBinary {
            spirv: None,
            metallib: None,
            entry_point: #func_name_str,
            stage: ::quanta::ShaderStage::ClosestHit,
        };

        pub fn #func_name() -> &'static ::quanta::ShaderBinary {
            &#binary_name
        }
    };
    expanded.into()
}

/// Mark a function as a miss shader.
///
/// Invoked when a ray does not intersect any geometry. Typically returns
/// a sky/environment color.
///
/// ```ignore
/// #[quanta::miss]
/// fn sky(ray: Ray) -> Vec4 {
///     let t = 0.5 * (ray.direction.y + 1.0);
///     Vec4::lerp(Vec4::splat(1.0), Vec4::new(0.5, 0.7, 1.0, 1.0), t)
/// }
/// ```
#[proc_macro_attribute]
pub fn miss(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let func_name = &func.sig.ident;
    let func_name_str = func_name.to_string();
    let binary_name = syn::Ident::new(
        &format!("{}_SHADER", func_name_str.to_uppercase()),
        func_name.span(),
    );

    let expanded = quote! {
        pub static #binary_name: ::quanta::ShaderBinary = ::quanta::ShaderBinary {
            spirv: None,
            metallib: None,
            entry_point: #func_name_str,
            stage: ::quanta::ShaderStage::Miss,
        };

        pub fn #func_name() -> &'static ::quanta::ShaderBinary {
            &#binary_name
        }
    };
    expanded.into()
}

/// Mark a struct as GPU-compatible.
///
/// Generates `#[repr(C)]`, `#[derive(Copy, Clone)]`, `GpuType` impl,
/// field metadata (`GPU_SIZE`, `GPU_FIELDS`), and MSL/WGSL struct declarations.
///
/// ```ignore
/// #[quanta::gpu_type]
/// struct Particle {
///     pos: [f32; 3],
///     vel: [f32; 3],
///     mass: f32,
/// }
/// ```
///
/// Generates:
/// - `Particle::GPU_SIZE` — byte size of the struct
/// - `Particle::GPU_FIELDS` — `&[(&str, &str, usize)]` of (name, type, byte_offset)
/// - `impl GpuType for Particle`
/// - `__QUANTA_GPU_TYPE_PARTICLE` — MSL struct declaration string
/// - `__QUANTA_GPU_TYPE_PARTICLE_WGSL` — WGSL struct declaration string
#[proc_macro_attribute]
pub fn gpu_type(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as syn::ItemStruct);
    match gpu_type::expand_gpu_type(&input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Parsed kernel attributes from `#[quanta::kernel(...)]`.
struct KernelAttrs {
    opt_level: u8,
    workgroup_size: [u32; 3],
}

impl Default for KernelAttrs {
    fn default() -> Self {
        Self {
            opt_level: 3,
            workgroup_size: [64, 1, 1],
        }
    }
}

/// Parse kernel attributes: `opt = "O2"`, `workgroup = [16, 16, 1]`, `jit`.
///
/// Supports:
/// - `#[quanta::kernel]`                           -> defaults
/// - `#[quanta::kernel(opt = "O2")]`               -> opt only
/// - `#[quanta::kernel(workgroup = [256])]`        -> [256, 1, 1]
/// - `#[quanta::kernel(workgroup = [16, 16])]`     -> [16, 16, 1]
/// - `#[quanta::kernel(workgroup = [16, 16, 1])]`  -> explicit 3D
/// - `#[quanta::kernel(workgroup = [256], opt = "O2")]` -> both
fn parse_kernel_attrs(attr: TokenStream) -> KernelAttrs {
    let mut attrs = KernelAttrs::default();

    if attr.is_empty() {
        return attrs;
    }

    // Try parsing as a punctuated list of name = value pairs.
    // We use syn to parse the token stream as comma-separated meta items.
    let parser = syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated;
    let parsed = match parser.parse(attr.clone()) {
        Ok(p) => p,
        Err(_) => {
            // Fall back: might be just `jit` or a single `opt = "O2"`.
            // Try single MetaNameValue parse for backward compat.
            if let Ok(nv) = syn::parse::<syn::MetaNameValue>(attr)
                && nv.path.is_ident("opt")
                && let Expr::Lit(expr_lit) = &nv.value
                && let Lit::Str(s) = &expr_lit.lit
            {
                attrs.opt_level = parse_opt_str(&s.value());
            }
            return attrs;
        }
    };

    for meta in &parsed {
        match meta {
            syn::Meta::NameValue(nv) if nv.path.is_ident("opt") => {
                if let Expr::Lit(expr_lit) = &nv.value
                    && let Lit::Str(s) = &expr_lit.lit
                {
                    attrs.opt_level = parse_opt_str(&s.value());
                }
            }
            syn::Meta::NameValue(nv) if nv.path.is_ident("workgroup") => {
                if let Some(wg) = parse_workgroup_expr(&nv.value) {
                    attrs.workgroup_size = wg;
                }
            }
            _ => {} // ignore `jit` and unknown attrs
        }
    }

    attrs
}

fn parse_opt_str(s: &str) -> u8 {
    match s {
        "O0" | "0" => 0,
        "O1" | "1" => 1,
        "O2" | "2" => 2,
        "O3" | "3" => 3,
        _ => 3,
    }
}

/// Parse `[256]`, `[16, 16]`, or `[16, 16, 1]` from an expression.
fn parse_workgroup_expr(expr: &Expr) -> Option<[u32; 3]> {
    if let Expr::Array(arr) = expr {
        let elems: Vec<u32> = arr
            .elems
            .iter()
            .filter_map(|e| {
                if let Expr::Lit(lit) = e
                    && let Lit::Int(int_lit) = &lit.lit
                {
                    return int_lit.base10_parse::<u32>().ok();
                }
                None
            })
            .collect();
        match elems.len() {
            1 => return Some([elems[0], 1, 1]),
            2 => return Some([elems[0], elems[1], 1]),
            3 => return Some([elems[0], elems[1], elems[2]]),
            _ => {}
        }
    }
    None
}
