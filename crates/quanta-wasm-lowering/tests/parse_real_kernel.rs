//! Integration test: parse the WASM emitted by the macro for a real
//! Quanta kernel (`examples/hello_quanta.rs`'s `vector_add`).
//!
//! `tests/hello_quanta.wasm` was generated via:
//!   cargo run -p quanta-cli -- wasm-experiment examples/hello_quanta.rs
//!   cp target/wasm-experiment/kernel.wasm \
//!      crates/quanta-wasm-lowering/tests/hello_quanta.wasm
//!
//! Refresh whenever the wasm-twin emitter changes shape. Until the
//! macro-side metadata emission lands, the test asserts only that
//! the parser sees what we expect: the `quark_id` import, the
//! `vector_add` export with three i32 params, and a non-empty
//! instruction stream.

use quanta_wasm_lowering::{
    ExportKind, FunctionKind, ImportKind, Module, RawInstr, WasmTy, find_kernel, parse_module,
};

const HELLO_QUANTA_WASM: &[u8] = include_bytes!("hello_quanta.wasm");

#[test]
fn parses_hello_quanta_kernel() {
    let module: Module = parse_module(HELLO_QUANTA_WASM).expect("parse module");

    let quark_id_import = module
        .imports
        .iter()
        .find(|i| i.name == "quark_id")
        .expect("quark_id import");
    assert_eq!(quark_id_import.module, "quanta");
    assert!(matches!(quark_id_import.kind, ImportKind::Function { .. }));

    let export = module
        .exports
        .iter()
        .find(|e| e.name == "vector_add")
        .expect("vector_add export");
    assert!(matches!(export.kind, ExportKind::Function { .. }));

    let (_idx, info) = find_kernel(&module, "vector_add").unwrap();
    let sig = &module.types[info.type_index as usize];
    assert_eq!(
        sig.params,
        vec![WasmTy::I32, WasmTy::I32, WasmTy::I32],
        "vector_add should take three i32 (pointer) params"
    );
    assert!(sig.results.is_empty(), "vector_add returns nothing");

    let body = match &info.kind {
        FunctionKind::Defined(b) => b,
        FunctionKind::Imported { .. } => panic!("vector_add should be defined, not imported"),
    };
    assert!(!body.instructions.is_empty(), "kernel body is non-empty");
    assert!(
        body.instructions
            .iter()
            .any(|op| matches!(op, RawInstr::F32Load { .. })),
        "expected at least one f32.load",
    );
    assert!(
        body.instructions
            .iter()
            .any(|op| matches!(op, RawInstr::F32Store { .. })),
        "expected at least one f32.store",
    );
    assert!(
        body.instructions
            .iter()
            .any(|op| matches!(op, RawInstr::Call(_))),
        "expected at least one call (quark_id)",
    );
}
