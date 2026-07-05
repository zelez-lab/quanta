//! Witness for the br_if-backedge + multi-level-br exit-tail shape.
//!
//! rustc produces this whenever a loop's exit continuation is not the
//! code lexically after the loop `end` — loop-unswitching (a
//! loop-invariant `if unit …` in the body becomes two loop copies
//! cross-jumped through a multi-level `br`), unroll epilogues, and the
//! div-by-zero panic guard all trigger it:
//!
//! ```wat
//! loop $L
//!   ...body...
//!   br_if 0 $L      ;; backedge: continue while cond
//!   br N            ;; exit tail: multi-level br out of the loop
//! end
//! ```
//!
//! The `br N` (and anything else after the backedge br_if) is on the
//! br_if's FALL-THROUGH path — it must only run when the backedge
//! does NOT fire. The old lowering emitted the exit tail sequentially
//! after `Branch { cond, then: [], else: [Break] }`, so the tail's
//! `flag = true; Break` ran unconditionally: every quark exited the
//! loop after one iteration (silent wrong result) and the emitted
//! SPIR-V had a branch to the loop continue target from outside the
//! loop construct (spirv-val-invalid).
//!
//! The fix records the backedge on the Loop frame and wraps the tail
//! at loop close: `Branch { cond, then: [], else: [tail…, Break] }`.
//!
//! The WAT mirrors rustc -O3 output for a loop-unswitched summation
//! kernel (`if unit == 1 { acc += k } else { acc += k / d }`); the
//! panic guard + call are elided by the panic-family elision, exactly
//! as in the real module.

use quanta_ir::{KernelOp, ScalarType};
use quanta_wasm_lowering::{ParamKind, ParamSlot, SideTable, lower};

fn side_table() -> SideTable {
    let mut params = vec![ParamSlot {
        wasm_index: 0,
        slot: 0,
        kind: ParamKind::BufferWrite,
        scalar: ScalarType::U32,
    }];
    params.extend((1u32..5).map(|i| ParamSlot {
        wasm_index: i,
        slot: i,
        kind: ParamKind::Scalar,
        scalar: ScalarType::U32,
    }));
    SideTable {
        kernel_name: "unswitch_sum".to_string(),
        params,
        workgroup_size: [64, 1, 1],
    }
}

/// rustc -O3 for:
///   while k < n { if unit == 1 { acc += k } else { acc += k / d }; k += s }
///   out[gid] = acc
/// Loop-unswitched into two loop copies; the first copy's exit is the
/// bug shape: `br_if 0` backedge followed by `br 2` crossing exit.
const UNSWITCH_SUM_WAT: &str = r#"
(module
  (import "quanta" "quark_id" (func $qid (result i32)))
  (memory 1)
  (func $unswitch_sum (export "unswitch_sum")
        (param i32 i32 i32 i32 i32) ;; out, n, d, s, unit
    (local i32 i32)                 ;; 5 = gid, 6 = k; local 4 (= param `unit`) is
                                    ;; recycled as `acc` by the register allocator
    call $qid
    local.set 5
    block ;; label = @1
      block ;; label = @2
        block ;; label = @3
          local.get 1
          br_if 0 (;@3;)
          i32.const 0
          local.set 4
          br 1 (;@2;)
        end
        block ;; label = @3
          local.get 4
          i32.const 1
          i32.ne
          br_if 0 (;@3;)
          i32.const 0
          local.set 4
          i32.const 0
          local.set 6
          loop ;; label = @4
            local.get 6
            local.get 4
            i32.add
            local.set 4
            local.get 6
            local.get 3
            i32.add
            local.tee 6
            local.get 1
            i32.lt_u
            br_if 0 (;@4;)   ;; backedge
            br 2 (;@2;)      ;; exit tail — THE SHAPE
          end
        end
        i32.const 0
        local.set 4
        i32.const 0
        local.set 6
        loop ;; label = @3
          local.get 6
          local.get 2
          i32.div_u
          local.get 4
          i32.add
          local.set 4
          local.get 6
          local.get 3
          i32.add
          local.tee 6
          local.get 1
          i32.lt_u
          br_if 0 (;@3;)     ;; plain last-instruction backedge
        end
      end
      local.get 0
      local.get 5
      i32.const 2
      i32.shl
      i32.add
      local.get 4
      i32.store
      return
    end
  )
)
"#;

fn find_loops(ops: &[KernelOp], out: &mut Vec<Vec<KernelOp>>) {
    for op in ops {
        match op {
            KernelOp::Loop { body, .. } => {
                out.push(body.clone());
                find_loops(body, out);
            }
            KernelOp::Branch {
                then_ops, else_ops, ..
            } => {
                find_loops(then_ops, out);
                find_loops(else_ops, out);
            }
            _ => {}
        }
    }
}

/// Any op at this nesting level that is an unconditional `Break` or a
/// bare exit-flag write NOT nested under a Branch — i.e. runs on every
/// iteration.
fn top_level_unconditional_break(body: &[KernelOp]) -> bool {
    body.iter().any(|op| matches!(op, KernelOp::Break))
}

#[test]
fn backedge_exit_tail_nests_inside_backedge_branch() {
    let wasm = wat::parse_str(UNSWITCH_SUM_WAT).expect("wat parse");
    let def = lower(&wasm, &side_table()).expect("lower unswitch_sum");
    quanta_ir::scope_check::scope_check(&def).expect("scope_check");

    let mut loops = Vec::new();
    find_loops(&def.body, &mut loops);
    assert_eq!(loops.len(), 2, "expected the two unswitched loop copies");

    for (i, body) in loops.iter().enumerate() {
        // No loop body may execute a Break unconditionally on the
        // fall-through path — that is the one-iteration bug.
        assert!(
            !top_level_unconditional_break(body),
            "loop {i}: unconditional Break at loop-body top level — the \
             exit tail escaped the backedge conditional; body = {body:#?}"
        );
        // The backedge must close the body: a Branch whose else arm
        // ends in Break (exit path), then nothing after it.
        let last = body.last().expect("loop body non-empty");
        let KernelOp::Branch {
            then_ops, else_ops, ..
        } = last
        else {
            panic!("loop {i}: body must end with the backedge Branch; got {last:?}");
        };
        assert!(
            then_ops.is_empty(),
            "loop {i}: backedge then-arm must be empty (fall-through = continue)"
        );
        assert!(
            matches!(else_ops.last(), Some(KernelOp::Break)),
            "loop {i}: backedge else-arm must end in Break; got {else_ops:#?}"
        );
    }

    // The first (unswitched, bug-shape) loop: its exit tail — the
    // crossing-exit flag write — must live INSIDE the backedge's else
    // arm, before the Break.
    let first = &loops[0];
    let KernelOp::Branch { else_ops, .. } = first.last().unwrap() else {
        unreachable!()
    };
    assert!(
        else_ops.len() >= 2,
        "first loop: exit tail (flag write) must be nested in the \
         backedge else-arm; else = {else_ops:#?}"
    );
}

// ── SPIR-V validity ──────────────────────────────────────────────────
//
// The old lowering also produced spirv-val-INVALID output for this
// shape ("branches to the loop continue target … not contained in the
// loop construct"): the unconditional tail's Break became a branch to
// the merge block from a position the structurizer had already routed
// to the continue target. Pipe the lowered KernelDef through the
// quanta-compiler binary and assert spirv-val is clean. Mirrors
// tests/validate_spirv.rs; self-skips when the compiler binary or
// spirv-val is missing.

const LLVM_PREFIX: &str = "/opt/homebrew/opt/llvm@22";
const SPIRV_VAL: &str = "/opt/homebrew/bin/spirv-val";

fn compiler_path() -> Option<std::path::PathBuf> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)?
        .to_path_buf();
    for dir in &["target/release", "target/debug"] {
        let p = root.join(dir).join("quanta-compiler");
        if p.exists() {
            return Some(p);
        }
    }
    None
}

#[test]
fn backedge_exit_tail_spirv_validates() {
    use std::io::Write as _;
    use std::process::{Command, Stdio};

    let Some(compiler) = compiler_path() else {
        eprintln!("skipping: quanta-compiler not built");
        return;
    };
    let spirv_val_ok = Command::new(SPIRV_VAL)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !spirv_val_ok {
        eprintln!("skipping: spirv-val not available");
        return;
    }

    let wasm = wat::parse_str(UNSWITCH_SUM_WAT).expect("wat parse");
    let def = lower(&wasm, &side_table()).expect("lower unswitch_sum");
    let input_bytes = quanta_ir::serialize_kernel(&def);

    let mut child = Command::new(&compiler)
        .args(["--targets", "spirv"])
        .env("LLVM_SYS_221_PREFIX", LLVM_PREFIX)
        .env("DYLD_LIBRARY_PATH", format!("{LLVM_PREFIX}/lib"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn quanta-compiler");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(&input_bytes)
        .expect("write kernel to compiler stdin");
    let output = child.wait_with_output().expect("compiler did not finish");
    assert!(
        output.status.success(),
        "compiler exited with error: {}",
        String::from_utf8_lossy(&output.stderr),
    );

    let co = quanta_ir::deserialize_output(&output.stdout).expect("deserialize CompilerOutput");
    let spirv = co.spirv.expect("compiler produced no SPIR-V output");
    // The LLVM SPIR-V backend may emit trailing metadata bytes that
    // break 4-byte alignment; truncate as tests/validate_spirv.rs does.
    let aligned = spirv.len() - (spirv.len() % 4);

    let path = std::env::temp_dir().join("quanta_backedge_exit_tail.spv");
    std::fs::write(&path, &spirv[..aligned]).unwrap();
    let val = Command::new(SPIRV_VAL)
        .arg(&path)
        .output()
        .expect("run spirv-val");
    let _ = std::fs::remove_file(&path);
    assert!(
        val.status.success(),
        "spirv-val rejected the lowered module:\n{}\n{}",
        String::from_utf8_lossy(&val.stdout),
        String::from_utf8_lossy(&val.stderr),
    );
}
