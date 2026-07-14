//! Implementation of `#[quanta::device]`.
//!
//! Marks a function as a GPU device function — callable from
//! `#[quanta::kernel]` bodies (and from plain CPU code, since the
//! function is also emitted unchanged for host use). The attribute:
//!
//! 1. Stores the function's source text in a process-wide registry,
//!    keyed by function name.
//! 2. Emits the function unchanged so it's also callable from CPU
//!    code (host-side reference, tests, doctests).
//!
//! When `#[quanta::kernel]` later expands, it walks the kernel body
//! for `Expr::Call` whose callee is a bare path (no `crate::` /
//! `super::` / `::` prefix), looks up each such name in this
//! registry, and prepends the device-function source into the
//! temporary wasm shell crate that rustc compiles. The WASM-route
//! lowering then handles the resulting `call $device_fn` instructions
//! through the existing helper-inlining path — at -O3 LLVM typically
//! folds them into straight-line code before they reach the lowerer.
//!
//! ## Ordering constraint
//!
//! Because all proc-macro invocations within one crate compilation
//! share this registry but the kernel macro expands eagerly, device
//! functions must appear *earlier in source order* than the kernels
//! that call them. In practice this matches how Rust users already
//! think: "define your helpers, then use them."
//!
//! Transitive calls (device-fn-calls-device-fn) are handled by
//! recursive discovery in `collect_device_sources_for` — each
//! callee's source is parsed and walked for further bare-path calls.

use proc_macro::TokenStream;
use quote::ToTokens;
use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};
use syn::visit::{self, Visit};
use syn::visit_mut::{self, VisitMut};
use syn::{Block, Expr, ExprCall, ExprPath, ItemFn, Path};

/// Process-wide registry of device-function sources, keyed by bare
/// function name. Populated by `#[quanta::device]`, read by
/// `#[quanta::kernel]`.
///
/// `OnceLock<Mutex<...>>` is the textbook stable-Rust way to share
/// mutable state across proc-macro attribute invocations within a
/// single proc-macro DSO process. Cargo invokes proc macros once per
/// crate compilation, so this scope is exactly "all macros that
/// expand in this crate's build."
fn registry() -> &'static Mutex<HashMap<String, String>> {
    static REG: OnceLock<Mutex<HashMap<String, String>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Register a device function's source text under its name. Called
/// from `expand_device` at attribute-expansion time.
fn register_device_source(name: &str, source: &str) {
    if let Ok(mut map) = registry().lock() {
        map.insert(name.to_string(), source.to_string());
    }
}

/// Look up a device function's source by name. Returns `None` if no
/// device function with that name has been registered yet — that's
/// usually a "called before defined" ordering issue.
fn lookup_device_source(name: &str) -> Option<String> {
    registry().lock().ok().and_then(|m| m.get(name).cloned())
}

/// Core implementation of `#[quanta::device]`.
///
/// Emits THREE things:
/// 1. The function unchanged — so plain CPU code can call it.
/// 2. A registry entry in the in-process source map — so a same-
///    crate `#[quanta::kernel]` can find the source at expansion
///    time (the existing path).
/// 3. A `#[macro_export] macro_rules! <name>_src` that expands to a
///    private copy of the function — so a *different* crate can
///    import it. The downstream crate writes
///    `crate_name::function_name_src!();` at file scope and the
///    function definition appears as if it were local. The kernel
///    macro then finds it via the same in-process registry, no
///    cross-process state needed.
pub(crate) fn expand_device(attr: TokenStream, func: ItemFn) -> TokenStream {
    use proc_macro2::TokenStream as TokenStream2;
    use quote::quote;

    let name = func.sig.ident.to_string();
    let source = func.to_token_stream().to_string();
    register_device_source(&name, &source);

    let fn_tokens: TokenStream2 = func.to_token_stream();

    // If the attribute carries `register_only`, the caller is the
    // auto-generated `_src!` macro re-registering this fn in the
    // downstream crate's macro process. Don't re-emit the `_src!`
    // macro itself (that would cause an `E0428` duplicate macro
    // definition on the original ident).
    let attr_str: String = attr.to_string();
    if attr_str.contains("register_only") {
        return fn_tokens.into();
    }

    // Crate-root override (`crate = <path>`). Baked into the emitted
    // `_src!` macro so a downstream crate that invokes `foo_src!()`
    // resolves `__device_host_stubs` and the re-registering
    // `device(register_only)` through the same crate the device fn was
    // defined against — not the facade (the whole point of the
    // override).
    let cp = crate::crate_path::from_attr_args(attr);
    let types_root = cp.types();
    let macro_root = cp.macros();

    let src_macro_ident = syn::Ident::new(&format!("{name}_src"), func.sig.ident.span());

    // The _src macro expands at the downstream call site to a
    // `const _: () = { ... }` anonymous block. Inside the block,
    // the device fn is re-declared with `#[quanta::device(register_only)]`
    // — registers in the downstream registry but does NOT re-emit
    // a `_src!` macro (which would collide if more than one
    // downstream site invokes the same `_src!`).
    //
    // The `use ::quanta::__device_host_stubs::*` brings host-side
    // stubs for every GPU intrinsic (`reduce_add_u32`,
    // `subgroup_size`, `shared_store_u32`, `barrier`, ...) into the
    // const block's scope so the spliced fn body name-resolves on
    // the host build of the downstream crate. Without this line,
    // any device fn whose body calls a Quanta intrinsic by bare
    // name fails to compile cross-crate with `E0425: cannot find
    // function`. The stubs are degenerate (reduce returns input,
    // shared memory is no-op) — they exist for name resolution,
    // not for execution; the GPU path emits real ops via the
    // wasm-shell `extern "C"` import block.
    let expanded = quote! {
        #fn_tokens

        /// Source-injection macro auto-generated by `#[quanta::device]`.
        ///
        /// Invoke it once at file scope in any crate that wants
        /// `#[quanta::kernel]` bodies to call this device function:
        ///
        /// ```ignore
        /// quanta_rand::name_src!();
        ///
        /// #[quanta::kernel]
        /// fn my_kernel(d: &MyData) { /* can now call name(...) */ }
        /// ```
        #[macro_export]
        #[doc(hidden)]
        macro_rules! #src_macro_ident {
            () => {
                const _: () = {
                    #[allow(unused_imports)]
                    use #types_root::__device_host_stubs::*;
                    #[allow(dead_code, non_snake_case)]
                    #[#macro_root::device(register_only)]
                    #fn_tokens
                };
            };
        }
    };
    expanded.into()
}

/// Visitor that collects every bare-path callee name in a syntax
/// tree — `foo(x)`, `bar()`. Skips qualified paths (`foo::bar(...)`,
/// `crate::baz(...)`, `Vec::new()`) because those are not local
/// device-function names. Also skips method calls (`x.foo()`) —
/// those follow a different resolution path.
struct BareCallCollector {
    names: HashSet<String>,
}

impl BareCallCollector {
    fn new() -> Self {
        Self {
            names: HashSet::new(),
        }
    }
}

impl<'ast> Visit<'ast> for BareCallCollector {
    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Expr::Path(ExprPath {
            path, qself: None, ..
        }) = node.func.as_ref()
            && path.leading_colon.is_none()
            && path.segments.len() == 1
        {
            let seg = &path.segments[0];
            if seg.arguments.is_empty() {
                self.names.insert(seg.ident.to_string());
            }
        }
        visit::visit_expr_call(self, node);
    }
}

/// Collect every device-function source the kernel transitively
/// needs.
///
/// Walks the kernel body for bare-path calls; for each name found in
/// the registry, recursively parses the device fn's source and walks
/// *its* body too. Returns the sources in stable order so the
/// generated wasm-shell crate has a stable hash for caching.
///
/// Unrecognized bare-call names (intrinsics like `quark_id`,
/// `sqrt_f32`, or the user's own functions that aren't marked
/// `#[quanta::device]`) are silently skipped — the wasm-shell rustc
/// will surface a clear "cannot find function" error if any of them
/// is actually missing.
pub(crate) fn collect_device_sources_for(kernel_body: &Block) -> Vec<String> {
    let mut collected: Vec<(String, String)> = Vec::new();
    let mut visited: HashSet<String> = HashSet::new();

    let mut stack: Vec<String> = {
        let mut v = BareCallCollector::new();
        v.visit_block(kernel_body);
        v.names.into_iter().collect()
    };
    // Sort for determinism — stable order in the generated wasm shell
    // means stable cache keys for `compile_kernel_to_wasm`.
    stack.sort();

    while let Some(name) = stack.pop() {
        if !visited.insert(name.clone()) {
            continue;
        }
        let Some(source) = lookup_device_source(&name) else {
            continue;
        };
        // Parse the device fn's source to discover its own bare-call
        // dependencies. If parsing fails we still keep the source —
        // rustc will surface any actual problem during the wasm
        // build.
        if let Ok(device_fn) = syn::parse_str::<ItemFn>(&source) {
            let mut inner = BareCallCollector::new();
            inner.visit_block(&device_fn.block);
            for dep in inner.names {
                if !visited.contains(&dep) {
                    stack.push(dep);
                }
            }
        }
        collected.push((name, source));
    }

    // Reverse so that deeper dependencies appear first. Rust `fn`s
    // can forward-reference each other within a module, so order
    // doesn't strictly matter for compilation, but stable order
    // keeps the wasm-shell hash stable.
    collected.reverse();
    collected.into_iter().map(|(_, s)| s).collect()
}

/// Rewriter that turns qualified `Expr::Call`s — `crate::path::foo(args)`
/// — into bare-name calls — `foo(args)`. Records each qualified path
/// it saw so the kernel macro can emit the matching `_src!()` macro
/// invocations as siblings.
///
/// "Qualified" here means a `Path` with at least 2 segments and no
/// type args on any of them. Common cases:
///   - `quanta_rand::philox4x32_10_first_u32_kernel(...)` → rewrite + record
///   - `crate::my_module::helper(...)` → rewrite + record (treated as cross-mod)
///   - `Vec::new()` → 2-segment but the receiver is a TYPE not a crate;
///     filtered by ensuring all earlier segments look like crate/mod
///     names (lower_snake_case heuristic).
///
/// Method calls (`x.foo()`) and bare-path calls (`foo()`) are left
/// alone — the bare-path collector handles same-crate device fns.
pub(crate) struct QualifiedDeviceCallRewriter {
    /// Qualified paths seen, in source order. Used to emit one
    /// `path_to_fn_src!();` per entry at the kernel-macro callsite.
    pub paths: Vec<Path>,
    /// Names already added (to dedupe).
    seen: HashSet<String>,
}

impl QualifiedDeviceCallRewriter {
    pub fn new() -> Self {
        Self {
            paths: Vec::new(),
            seen: HashSet::new(),
        }
    }
}

impl Default for QualifiedDeviceCallRewriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Cheap heuristic: a path segment looks like a *module/crate*
/// (rather than a type) if its ident is lower_snake_case.
fn looks_like_mod_or_crate_segment(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

impl VisitMut for QualifiedDeviceCallRewriter {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        // Recurse first so nested calls get rewritten too.
        visit_mut::visit_expr_mut(self, expr);

        let Expr::Call(call) = expr else {
            return;
        };
        let Expr::Path(ExprPath {
            path,
            qself: None,
            attrs,
        }) = call.func.as_ref()
        else {
            return;
        };
        if !attrs.is_empty() || path.leading_colon.is_some() {
            return;
        }
        if path.segments.len() < 2 {
            return;
        }
        // The final segment must have no type args and must look
        // like a free function (heuristic: any-case ident, not
        // PascalCase like `Vec::new` where the second-to-last seg
        // is a TYPE).
        let last = path.segments.last().unwrap();
        if !last.arguments.is_empty() {
            return;
        }
        // All non-final segments must look like crate/mod names
        // (lower_snake_case). This rules out `Vec::new`,
        // `String::from`, `Self::foo`, etc.
        for seg in path.segments.iter().take(path.segments.len() - 1) {
            if !looks_like_mod_or_crate_segment(&seg.ident.to_string()) {
                return;
            }
        }

        // Looks like a qualified call to a free fn in another mod
        // or crate. Record the qualified path and rewrite the call
        // to use just the final segment.
        let qualified_str = path
            .segments
            .iter()
            .map(|s| s.ident.to_string())
            .collect::<Vec<_>>()
            .join("::");
        if self.seen.insert(qualified_str) {
            self.paths.push(path.clone());
        }
        // Rewrite: replace path with just the final segment (bare).
        let bare = last.ident.clone();
        let bare_path: Path = syn::parse_quote!(#bare);
        *call.func = Expr::Path(ExprPath {
            path: bare_path,
            qself: None,
            attrs: Vec::new(),
        });
    }
}
