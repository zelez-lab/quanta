//! The browser glue, embedded in the crate.
//!
//! The wasm32/WebGPU face imports ~80 `env` functions that only the JS glue
//! provides (`web/src/*.ts`, compiled to `web/dist/*.js` — Quanta's
//! hand-rolled ABI; no wasm-bindgen, no wgpu). A consumer that depends on
//! Quanta through cargo alone — no repo checkout, no `quanta-cli`, no Node —
//! still needs those files to serve a page, so the compiled glue is COMMITTED
//! (`web/dist/` is tracked; web-smoke CI rebuilds it from the TypeScript and
//! fails on drift) and embedded here.
//!
//! The glue is a plain ES-module tree, not a single bundle: [`ENTRY`]
//! (`quanta.js`) imports its sibling modules by relative path. To ship it,
//! write every `(path, contents)` pair in [`FILES`] into one directory —
//! preserving the `generated/` subdirectory — next to the `.wasm` binary,
//! and load the entry from the page:
//!
//! ```js
//! import { instantiate } from "./quanta.js";
//! const mod = await instantiate("./app.wasm");
//! const canvas = mod.registerCanvas(document.getElementById("view"));
//! ```
//!
//! The table is a plain `const` on every target and behind no feature gate —
//! a build tool (a bundler CLI, a build.rs, an asset pipeline) reads it from
//! a NATIVE build of this crate without pulling the `webgpu` feature or any
//! GPU backend into its own graph.

/// Every file of the browser glue, as `(relative path, contents)` — the
/// exact bytes `quanta build web` stages next to each example's wasm.
pub const FILES: &[(&str, &str)] = &[
    ("quanta.js", include_str!("../web/dist/quanta.js")),
    ("webgpu.js", include_str!("../web/dist/webgpu.js")),
    (
        "webgpu-types.js",
        include_str!("../web/dist/webgpu-types.js"),
    ),
    ("codes.js", include_str!("../web/dist/codes.js")),
    ("handles.js", include_str!("../web/dist/handles.js")),
    ("strings.js", include_str!("../web/dist/strings.js")),
    ("tasks.js", include_str!("../web/dist/tasks.js")),
    (
        "generated/codes.js",
        include_str!("../web/dist/generated/codes.js"),
    ),
];

/// The entry module's relative path within [`FILES`] — what the page
/// imports (`instantiate`, `registerCanvas`, `makeImports` live there or
/// are re-exported through it).
pub const ENTRY: &str = "quanta.js";

#[cfg(test)]
mod tests {
    use super::*;

    /// The table carries the entry module and every sibling it imports —
    /// a `./name.js` import in any embedded file must resolve within the
    /// table, or a consumer writing `FILES` to disk ships a broken module
    /// graph. (Allocation-free: the crate is `no_std`. Only top-level
    /// modules import; the one subdirectory file is asserted import-free
    /// so root-relative resolution stays sound.)
    #[test]
    fn glue_module_graph_is_closed() {
        assert!(FILES.iter().any(|(p, _)| *p == ENTRY));
        for (path, contents) in FILES {
            assert!(!contents.is_empty(), "{path} embedded empty");
            for line in contents.lines() {
                // Both import forms: `from "./x.js"` and the side-effect
                // `import "./x.js"` (webgpu-types.js is loaded that way).
                let rest = match (
                    line.split("from \"./").nth(1),
                    line.split("import \"./").nth(1),
                ) {
                    (Some(r), _) | (None, Some(r)) => r,
                    (None, None) => continue,
                };
                let Some(import) = rest.split('"').next() else {
                    continue;
                };
                assert!(
                    !path.contains('/'),
                    "{path} lives in a subdirectory but imports ./{import}; \
                     the root-relative resolution below no longer holds"
                );
                assert!(
                    FILES.iter().any(|(p, _)| *p == import),
                    "{path} imports ./{import} which is not in FILES"
                );
            }
        }
    }
}
