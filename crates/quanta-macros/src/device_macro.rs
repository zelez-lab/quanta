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
use syn::{Block, Expr, ExprCall, ExprPath, ItemFn};

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
pub(crate) fn expand_device(func: ItemFn) -> TokenStream {
    use proc_macro2::TokenStream as TokenStream2;
    use quote::quote;

    let name = func.sig.ident.to_string();
    let source = func.to_token_stream().to_string();
    register_device_source(&name, &source);

    let fn_tokens: TokenStream2 = func.to_token_stream();
    let src_macro_ident = syn::Ident::new(&format!("{name}_src"), func.sig.ident.span());

    // The _src macro expands at the downstream call site to a
    // `const _: () = { ... }` anonymous block. Inside the block:
    //
    //   (a) the device fn is re-declared with `#[quanta::device]`,
    //       so its source registers in the downstream crate's
    //       macro process and downstream kernels can find it,
    //   (b) the block is anonymous (`const _`) — neither the
    //       constant nor the fn it contains is reachable from
    //       outside, so we don't leak any new identifier into the
    //       user's namespace.
    //
    // The original v0 approach used `mod __quanta_device_src_<name>`
    // which DID leak a mod name into the user's crate. `const _`
    // is the standard Rust idiom for "run this item-level code in
    // a fresh anonymous scope" — same as how serde / tonic / etc.
    // emit their auto-generated impls.
    //
    // The inner fn shadows any same-named import the downstream
    // crate might have (e.g. `use quanta_rand::foo;`). That's fine
    // because the const block creates a fresh scope: `foo` inside
    // the block is the device-fn definition; `foo` outside the
    // block is whatever the user imported.
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
                    #[allow(dead_code, non_snake_case)]
                    #[quanta::device]
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
