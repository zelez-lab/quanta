//! `quanta codegen webgpu` — drive the `quanta-codegen` crate.
//!
//! Thin shim: locate the workspace root, hand `web/webgpu.idl` to
//! `quanta_codegen::generate`, and let it write the outputs.

use crate::Result;
use crate::workspace;

pub fn webgpu() -> Result<()> {
    let root = workspace::root()?;
    let idl = root.join("web/webgpu.idl");
    if !idl.is_file() {
        return Err(format!(
            "web/webgpu.idl not found at {} — vendor it first via\n  curl -sSfL https://gpuweb.github.io/gpuweb/webgpu.idl -o web/webgpu.idl",
            idl.display()
        )
        .into());
    }
    quanta_codegen::generate(&idl, &root).map_err(|e| format!("quanta codegen webgpu: {e}").into())
}
