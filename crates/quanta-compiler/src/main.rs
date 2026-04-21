//! Quanta kernel compiler — reads KernelDef, emits GPU code.
//!
//! Usage:
//!   echo '<bincode>' | quanta-compiler --targets nvptx,amdgpu
//!   quanta-compiler --test-ir    # test with a built-in vector_add kernel

mod emit_msl;
mod emit_spirv;
mod emit_wgsl;
mod rustc_compile;
mod targets;
mod to_llvm;

use quanta_ir::*;
use targets::GpuTarget;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--test-ir") {
        test_ir();
        return;
    }

    if args.iter().any(|a| a == "--test-ptx") {
        test_ptx();
        return;
    }

    if args.iter().any(|a| a == "--test-amd") {
        test_amd();
        return;
    }

    if args.iter().any(|a| a == "--test-complex") {
        test_complex();
        return;
    }

    if args.iter().any(|a| a == "--test-spirv") {
        test_spirv();
        return;
    }

    if args.iter().any(|a| a == "--test-rustc") {
        test_rustc();
        return;
    }

    // Normal mode: read KernelDef from stdin, emit results to stdout
    let mut input = Vec::new();
    std::io::Read::read_to_end(&mut std::io::stdin(), &mut input).unwrap();
    let kernel: KernelDef = quanta_ir::deserialize_kernel(&input).unwrap();

    let targets = parse_targets(&args);
    let mut output = CompilerOutput {
        amd: None,
        nvidia: None,
        spirv: None,
        metallib: None,
        msl: None,
        wgsl: None,
        llvm_ir: None,
    };

    // Always generate MSL + WGSL (lightweight, from KernelOps — no LLVM needed)
    output.msl = emit_msl::emit(&kernel).ok();
    output.wgsl = emit_wgsl::emit(&kernel).ok();

    // Compile MSL → metallib via xcrun (if available)
    if let Some(ref msl) = output.msl {
        output.metallib = compile_msl_to_metallib(msl);
    }

    // Emit Vulkan SPIR-V directly from KernelOps (Shader capability, GLCompute).
    // Our own emitter — no naga, no LLVM spirv64 backend.
    output.spirv = emit_spirv::emit(&kernel).ok();

    // Generate LLVM-compiled targets
    // Strategy: try rustc path first (handles ALL Rust), fall back to KernelOp path
    let use_rustc = kernel.body_source.is_some();

    for target in &targets {
        let result = if use_rustc {
            // rustc path: Rust source → rustc → LLVM IR → retarget
            // Then we'd feed the retargeted IR to our LLVM pipeline.
            // For now, use rustc to get verified IR, then fall back to KernelOp path
            // for actual binary emission (until we wire up IR → inkwell → binary).
            to_llvm::compile_to_binary(&kernel, *target)
        } else {
            to_llvm::compile_to_binary(&kernel, *target)
        };

        match result {
            Ok(binary) => match target {
                GpuTarget::Nvptx => output.nvidia = Some(binary),
                // Don't overwrite direct/naga SPIR-V with LLVM's OpenCL-style SPIR-V
                GpuTarget::Spirv if output.spirv.is_some() => {}
                GpuTarget::Amdgpu => output.amd = Some(binary),
                GpuTarget::Spirv => output.spirv = Some(binary),
            },
            Err(e) => {
                eprintln!("Error compiling for {:?}: {}", target, e);
            }
        }
    }

    let out_bytes = quanta_ir::serialize_output(&output);
    std::io::Write::write_all(&mut std::io::stdout(), &out_bytes).unwrap();
}

/// Compile MSL source to metallib binary via xcrun metal + xcrun metallib.
/// Returns None if xcrun is not available (e.g., cross-compiling from Linux).
fn compile_msl_to_metallib(msl_source: &str) -> Option<Vec<u8>> {
    let tmp_dir = std::env::temp_dir().join("quanta_metal");
    std::fs::create_dir_all(&tmp_dir).ok()?;

    let msl_path = tmp_dir.join("kernel.metal");
    let air_path = tmp_dir.join("kernel.air");
    let lib_path = tmp_dir.join("kernel.metallib");

    std::fs::write(&msl_path, msl_source).ok()?;

    // MSL → AIR
    let air_result = std::process::Command::new("xcrun")
        .args(["metal", "-c", "-target", "air64-apple-macos14.0.0"])
        .arg(&msl_path)
        .arg("-o")
        .arg(&air_path)
        .stderr(std::process::Stdio::piped())
        .output();

    match air_result {
        Ok(o) if o.status.success() => o,
        Ok(o) => {
            let err = String::from_utf8_lossy(&o.stderr);
            eprintln!("[quanta] xcrun metal failed: {}", err);
            return None;
        }
        Err(_) => return None, // xcrun not found
    };

    // AIR → metallib
    let lib_result = std::process::Command::new("xcrun")
        .args(["metallib"])
        .arg(&air_path)
        .arg("-o")
        .arg(&lib_path)
        .stderr(std::process::Stdio::piped())
        .output();

    match lib_result {
        Ok(o) if o.status.success() => std::fs::read(&lib_path).ok(),
        Ok(o) => {
            let err = String::from_utf8_lossy(&o.stderr);
            eprintln!("[quanta] xcrun metallib failed: {}", err);
            None
        }
        Err(_) => None,
    }
}

fn parse_targets(args: &[String]) -> Vec<GpuTarget> {
    for (i, arg) in args.iter().enumerate() {
        if arg == "--targets"
            && let Some(list) = args.get(i + 1)
        {
            return list
                .split(',')
                .filter_map(|s| match s.trim() {
                    "nvptx" => Some(GpuTarget::Nvptx),
                    "amdgpu" => Some(GpuTarget::Amdgpu),
                    "spirv" => Some(GpuTarget::Spirv),
                    _ => None,
                })
                .collect();
        }
    }
    // Default: both
    vec![GpuTarget::Nvptx, GpuTarget::Amdgpu]
}

fn make_test_kernel() -> KernelDef {
    KernelDef {
        name: "vector_add".to_string(),
        params: vec![
            KernelParam::FieldRead {
                name: "a".to_string(),
                slot: 0,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldRead {
                name: "b".to_string(),
                slot: 1,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldWrite {
                name: "result".to_string(),
                slot: 2,
                scalar_type: ScalarType::F32,
            },
        ],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Load {
                dst: Reg(1),
                field: 0,
                index: Reg(0),
                ty: ScalarType::F32,
            },
            KernelOp::Load {
                dst: Reg(2),
                field: 1,
                index: Reg(0),
                ty: ScalarType::F32,
            },
            KernelOp::BinOp {
                dst: Reg(3),
                a: Reg(1),
                b: Reg(2),
                op: BinOp::Add,
                ty: ScalarType::F32,
            },
            KernelOp::Store {
                field: 2,
                index: Reg(0),
                src: Reg(3),
                ty: ScalarType::F32,
            },
        ],
        body_source: None,
        next_reg: 4,
        opt_level: 3,
        device_sources: Vec::new(),
    }
}

/// Test: compile vector_add to PTX and print it.
fn test_ptx() {
    let kernel = make_test_kernel();
    println!("=== Compiling vector_add to NVIDIA PTX ===\n");
    match to_llvm::compile_to_binary(&kernel, GpuTarget::Nvptx) {
        Ok(ptx) => {
            let ptx_text = String::from_utf8_lossy(&ptx);
            println!("{}", ptx_text);
            println!("\n=== PTX size: {} bytes ===", ptx.len());
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}

/// Test: compile vector_add to AMD ELF and show stats.
fn test_amd() {
    let kernel = make_test_kernel();
    println!("=== Compiling vector_add to AMD GCN ELF ===\n");
    match to_llvm::compile_to_binary(&kernel, GpuTarget::Amdgpu) {
        Ok(elf) => {
            println!("ELF binary size: {} bytes", elf.len());
            // Print first few bytes as hex
            print!("Header: ");
            for b in elf.iter().take(16) {
                print!("{:02x} ", b);
            }
            println!();
            // Check ELF magic
            if elf.len() >= 4 && elf[0..4] == [0x7f, b'E', b'L', b'F'] {
                println!("✓ Valid ELF binary");
            } else {
                println!("✗ Not an ELF binary");
            }
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}

/// Test: compile a complex neuron activation kernel to PTX.
fn test_complex() {
    // Neuron activation: accumulate 16 weighted signals, threshold, decay
    let kernel = KernelDef {
        name: "neuron_activate".to_string(),
        params: vec![
            KernelParam::FieldRead {
                name: "potentials".to_string(),
                slot: 0,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldRead {
                name: "signals".to_string(),
                slot: 1,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldRead {
                name: "weights".to_string(),
                slot: 2,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldRead {
                name: "thresholds".to_string(),
                slot: 3,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldWrite {
                name: "output".to_string(),
                slot: 4,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldWrite {
                name: "fired".to_string(),
                slot: 5,
                scalar_type: ScalarType::U32,
            },
        ],
        body: {
            let mut ops = Vec::new();
            // let i = quark_id()
            ops.push(KernelOp::QuarkId { dst: Reg(0) });
            // let p = potentials[i]
            ops.push(KernelOp::Load {
                dst: Reg(1),
                field: 0,
                index: Reg(0),
                ty: ScalarType::F32,
            });
            // const 16
            ops.push(KernelOp::Const {
                dst: Reg(2),
                value: ConstValue::I32(16),
            });
            // base = i * 16
            ops.push(KernelOp::Cast {
                dst: Reg(3),
                src: Reg(0),
                from: ScalarType::U32,
                to: ScalarType::I32,
            });
            ops.push(KernelOp::BinOp {
                dst: Reg(4),
                a: Reg(3),
                b: Reg(2),
                op: BinOp::Mul,
                ty: ScalarType::I32,
            });
            // loop: for j in 0..16 { p += signals[base+j] * weights[base+j] }
            ops.push(KernelOp::Cast {
                dst: Reg(20),
                src: Reg(2),
                from: ScalarType::I32,
                to: ScalarType::U32,
            });
            ops.push(KernelOp::Loop {
                count: Reg(20),
                iter_reg: Reg(5),
                body: {
                    vec![
                        // idx = base + j
                        KernelOp::Cast {
                            dst: Reg(6),
                            src: Reg(5),
                            from: ScalarType::U32,
                            to: ScalarType::I32,
                        },
                        KernelOp::BinOp {
                            dst: Reg(7),
                            a: Reg(4),
                            b: Reg(6),
                            op: BinOp::Add,
                            ty: ScalarType::I32,
                        },
                        KernelOp::Cast {
                            dst: Reg(8),
                            src: Reg(7),
                            from: ScalarType::I32,
                            to: ScalarType::U32,
                        },
                        // s = signals[idx]
                        KernelOp::Load {
                            dst: Reg(9),
                            field: 1,
                            index: Reg(8),
                            ty: ScalarType::F32,
                        },
                        // w = weights[idx]
                        KernelOp::Load {
                            dst: Reg(10),
                            field: 2,
                            index: Reg(8),
                            ty: ScalarType::F32,
                        },
                        // p += s * w
                        KernelOp::BinOp {
                            dst: Reg(11),
                            a: Reg(9),
                            b: Reg(10),
                            op: BinOp::Mul,
                            ty: ScalarType::F32,
                        },
                        KernelOp::BinOp {
                            dst: Reg(1),
                            a: Reg(1),
                            b: Reg(11),
                            op: BinOp::Add,
                            ty: ScalarType::F32,
                        },
                    ]
                },
            });
            // threshold = thresholds[i]
            ops.push(KernelOp::Load {
                dst: Reg(12),
                field: 3,
                index: Reg(0),
                ty: ScalarType::F32,
            });
            // if p > threshold
            ops.push(KernelOp::Cmp {
                dst: Reg(13),
                a: Reg(1),
                b: Reg(12),
                op: CmpOp::Gt,
                ty: ScalarType::F32,
            });
            ops.push(KernelOp::Branch {
                cond: Reg(13),
                then_ops: vec![
                    // output[i] = p
                    KernelOp::Store {
                        field: 4,
                        index: Reg(0),
                        src: Reg(1),
                        ty: ScalarType::F32,
                    },
                    // fired[i] = 1
                    KernelOp::Const {
                        dst: Reg(14),
                        value: ConstValue::U32(1),
                    },
                    KernelOp::Store {
                        field: 5,
                        index: Reg(0),
                        src: Reg(14),
                        ty: ScalarType::U32,
                    },
                ],
                else_ops: vec![
                    // output[i] = p * 0.99
                    KernelOp::Const {
                        dst: Reg(15),
                        value: ConstValue::F32(0.99),
                    },
                    KernelOp::BinOp {
                        dst: Reg(16),
                        a: Reg(1),
                        b: Reg(15),
                        op: BinOp::Mul,
                        ty: ScalarType::F32,
                    },
                    KernelOp::Store {
                        field: 4,
                        index: Reg(0),
                        src: Reg(16),
                        ty: ScalarType::F32,
                    },
                    // fired[i] = 0
                    KernelOp::Const {
                        dst: Reg(17),
                        value: ConstValue::U32(0),
                    },
                    KernelOp::Store {
                        field: 5,
                        index: Reg(0),
                        src: Reg(17),
                        ty: ScalarType::U32,
                    },
                ],
            });
            ops
        },
        body_source: None,
        next_reg: 21,
        opt_level: 3,
        device_sources: Vec::new(),
    };

    println!("=== NVIDIA PTX (neuron_activate, O3) ===\n");
    match to_llvm::compile_to_binary(&kernel, GpuTarget::Nvptx) {
        Ok(ptx) => {
            println!("{}", String::from_utf8_lossy(&ptx));
            println!("=== PTX size: {} bytes ===", ptx.len());
        }
        Err(e) => eprintln!("Error: {}", e),
    }

    println!("\n=== AMD ELF (neuron_activate, O3) ===\n");
    match to_llvm::compile_to_binary(&kernel, GpuTarget::Amdgpu) {
        Ok(elf) => {
            println!("ELF size: {} bytes", elf.len());
            if elf.len() >= 4 && elf[0..4] == [0x7f, b'E', b'L', b'F'] {
                println!("✓ Valid ELF");
            }
        }
        Err(e) => eprintln!("Error: {}", e),
    }

    println!("\n=== LLVM IR (neuron_activate) ===\n");
    match to_llvm::compile_to_llvm_ir(&kernel, GpuTarget::Nvptx) {
        Ok(ir) => println!("{}", ir),
        Err(e) => eprintln!("Error: {}", e),
    }

    println!("\n=== Direct SPIR-V (neuron_activate) ===\n");
    match emit_spirv::emit(&kernel) {
        Ok(spirv) => {
            println!("SPIR-V binary size: {} bytes", spirv.len());
            let path = std::env::temp_dir().join("quanta_complex.spv");
            if std::fs::write(&path, &spirv).is_ok() {
                println!("Written to: {}", path.display());
                let val = std::process::Command::new("spirv-val").arg(&path).output();
                match val {
                    Ok(o) if o.status.success() => println!("spirv-val: PASS"),
                    Ok(o) => {
                        let err = String::from_utf8_lossy(&o.stderr);
                        println!("spirv-val: FAIL\n{}", err);
                    }
                    Err(_) => println!("spirv-val not found"),
                }
            }
        }
        Err(e) => eprintln!("Direct SPIR-V error: {}", e),
    }
}

/// Test: compile vector_add to LLVM IR text and print it.
fn test_ir() {
    let kernel = make_test_kernel();

    println!("=== NVPTX LLVM IR ===");
    match to_llvm::compile_to_llvm_ir(&kernel, GpuTarget::Nvptx) {
        Ok(ir) => println!("{}", ir),
        Err(e) => eprintln!("NVPTX error: {}", e),
    }

    println!("\n=== AMDGPU LLVM IR ===");
    match to_llvm::compile_to_llvm_ir(&kernel, GpuTarget::Amdgpu) {
        Ok(ir) => println!("{}", ir),
        Err(e) => eprintln!("AMDGPU error: {}", e),
    }

    println!("\n=== SPIR-V LLVM IR ===");
    match to_llvm::compile_to_llvm_ir(&kernel, GpuTarget::Spirv) {
        Ok(ir) => println!("{}", ir),
        Err(e) => eprintln!("SPIR-V error: {}", e),
    }
}

/// Test: compile vector_add to SPIR-V binary.
fn test_spirv() {
    let kernel = make_test_kernel();

    println!("=== Direct SPIR-V emitter (Vulkan compute) ===\n");
    match emit_spirv::emit(&kernel) {
        Ok(spirv) => {
            println!("SPIR-V binary size: {} bytes", spirv.len());
            if spirv.len() >= 4 {
                print!("Header: ");
                for b in spirv.iter().take(20) {
                    print!("{:02x} ", b);
                }
                println!();
                let magic = u32::from_le_bytes([spirv[0], spirv[1], spirv[2], spirv[3]]);
                if magic == 0x07230203 {
                    println!("Valid SPIR-V magic");
                }
            }
            // Write to tmp for spirv-val
            let path = std::env::temp_dir().join("quanta_test.spv");
            if std::fs::write(&path, &spirv).is_ok() {
                println!("Written to: {}", path.display());
                let val = std::process::Command::new("spirv-val").arg(&path).output();
                match val {
                    Ok(o) if o.status.success() => println!("spirv-val: PASS"),
                    Ok(o) => {
                        let err = String::from_utf8_lossy(&o.stderr);
                        println!("spirv-val: FAIL\n{}", err);
                    }
                    Err(_) => println!("spirv-val not found"),
                }
            }
        }
        Err(e) => eprintln!("Direct SPIR-V error: {}", e),
    }

    println!("\n=== SPIR-V LLVM IR (reference) ===\n");
    match to_llvm::compile_to_llvm_ir(&kernel, GpuTarget::Spirv) {
        Ok(ir) => println!("{}", ir),
        Err(e) => eprintln!("SPIR-V IR error: {}", e),
    }
}

/// Test: compile Rust source via rustc → LLVM IR.
fn test_rustc() {
    let params = vec![
        KernelParam::FieldRead {
            name: "a".into(),
            slot: 0,
            scalar_type: ScalarType::F32,
        },
        KernelParam::FieldRead {
            name: "b".into(),
            slot: 1,
            scalar_type: ScalarType::F32,
        },
        KernelParam::FieldWrite {
            name: "result".into(),
            slot: 2,
            scalar_type: ScalarType::F32,
        },
    ];

    let body = r#"
    let i = quark_id();
    result[i] = a[i] + b[i];
"#;

    println!("=== Rust source → rustc → LLVM IR ===\n");
    match rustc_compile::rust_to_llvm_ir("vector_add", &params, body) {
        Ok(ir) => {
            println!("{}", ir);
            println!("\n=== Success: {} bytes of LLVM IR ===", ir.len());
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
