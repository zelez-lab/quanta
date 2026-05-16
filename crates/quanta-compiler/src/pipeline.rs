//! Kernel compilation pipeline — stdin → compile → stdout.
//!
//! Handles the normal compilation mode (read KernelDef from stdin,
//! emit CompilerOutput to stdout) and LLVM subprocess mode.

use quanta_ir::*;

use crate::targets::GpuTarget;
use crate::{emit_llvm, emit_msl, emit_spirv, emit_wgsl, metallib};

/// Parse `--targets nvptx,amdgpu` from CLI args.
pub fn parse_targets(args: &[String]) -> Vec<GpuTarget> {
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

/// LLVM subprocess mode: compile a single target, write raw binary to stdout.
///
/// Used by the parent process to isolate LLVM fatal errors (abort on fsin etc.)
pub fn llvm_only(target: GpuTarget) {
    let mut input = Vec::new();
    std::io::Read::read_to_end(&mut std::io::stdin(), &mut input).unwrap();
    let kernel: KernelDef = quanta_ir::deserialize_kernel(&input).unwrap();
    match emit_llvm::compile_to_binary(&kernel, target) {
        Ok(binary) => {
            std::io::Write::write_all(&mut std::io::stdout(), &binary).unwrap();
        }
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    }
}

/// Normal compilation mode: read KernelDef from stdin, emit all targets to stdout.
pub fn compile_kernel(args: &[String]) {
    let mut input = Vec::new();
    std::io::Read::read_to_end(&mut std::io::stdin(), &mut input).unwrap();
    let kernel: KernelDef = quanta_ir::deserialize_kernel(&input).unwrap();

    let targets = parse_targets(args);
    let mut output = CompilerOutput {
        amd: None,
        nvidia: None,
        spirv: None,
        metallib: None,
        wgsl: None,
    };

    // Step 082 Layer 4: gate each backend's emission on the IR
    // validator. Kernels using NotSupported types get a clean skip
    // line on stderr instead of being passed to an emitter that
    // produces invalid backend text (e.g. F64 -> MSL "double",
    // which xcrun rejects with no recovery).
    let metal_report = quanta_ir::validate::validate_for(&quanta_ir::caps::METAL, &kernel);
    if metal_report.is_ok() {
        if let Ok(msl) = emit_msl::emit(&kernel) {
            match metallib::compile_msl_to_metallib(&msl) {
                Ok(bytes) => output.metallib = bytes,
                Err(e) => eprintln!("[quanta] metallib error: {}", e),
            }
        }
    } else {
        eprintln!("[quanta] skipping metal emission: {}", metal_report);
    }

    // Emit WGSL source for WebGPU.
    let webgpu_report = quanta_ir::validate::validate_for(&quanta_ir::caps::WEBGPU, &kernel);
    if webgpu_report.is_ok() {
        match emit_wgsl::emit(&kernel) {
            Ok(wgsl) => output.wgsl = Some(wgsl),
            Err(e) => eprintln!("[quanta] WGSL emitter error: {}", e),
        }
    } else {
        eprintln!("[quanta] skipping wgsl emission: {}", webgpu_report);
    }

    // Emit Vulkan SPIR-V directly from KernelOps (Shader capability, GLCompute).
    let vulkan_report = quanta_ir::validate::validate_for(&quanta_ir::caps::VULKAN, &kernel);
    if vulkan_report.is_ok() {
        match emit_spirv::emit(&kernel) {
            Ok(spirv) => output.spirv = Some(spirv),
            Err(e) => eprintln!("[quanta] SPIR-V emitter error: {}", e),
        }
    } else {
        eprintln!("[quanta] skipping spirv emission: {}", vulkan_report);
    }

    // LLVM compilation for PTX/GCN — run in subprocess to survive fatal errors.
    // LLVM's error handler calls abort() on unsupported ops (e.g. fsin on SPIR-V target),
    // which would kill this process before metallib + SPIR-V are written to stdout.
    let self_exe = std::env::current_exe().unwrap_or_default();
    for target in &targets {
        let target_name = match target {
            GpuTarget::Nvptx => "nvptx",
            GpuTarget::Amdgpu => "amdgpu",
            GpuTarget::Spirv => continue, // already emitted above
        };

        let child = std::process::Command::new(&self_exe)
            .arg("--llvm-only")
            .arg(target_name)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn();

        if let Ok(mut child) = child {
            {
                if let Some(ref mut stdin) = child.stdin {
                    let _ = std::io::Write::write_all(stdin, &input);
                }
            }
            child.stdin.take(); // close stdin so child sees EOF
            if let Ok(result) = child.wait_with_output()
                && result.status.success()
                && !result.stdout.is_empty()
            {
                match target {
                    GpuTarget::Nvptx => output.nvidia = Some(result.stdout),
                    GpuTarget::Amdgpu => output.amd = Some(result.stdout),
                    GpuTarget::Spirv => {}
                }
            }
        }
    }

    let out_bytes = quanta_ir::serialize_output(&output);
    std::io::Write::write_all(&mut std::io::stdout(), &out_bytes).unwrap();
}

/// Build a test vector_add kernel definition.
pub fn make_test_kernel() -> KernelDef {
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
        device_functions: Vec::new(),
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

/// Test: compile vector_add to PTX and print it.
///
/// Banner + size footer go to stderr so stdout is pure PTX,
/// safe to redirect to a file and feed straight into ptxas.
pub fn test_ptx() {
    let kernel = make_test_kernel();
    eprintln!("=== Compiling vector_add to NVIDIA PTX ===");
    match emit_llvm::compile_to_binary(&kernel, GpuTarget::Nvptx) {
        Ok(ptx) => {
            let ptx_text = String::from_utf8_lossy(&ptx);
            print!("{}", ptx_text);
            eprintln!("=== PTX size: {} bytes ===", ptx.len());
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}

/// Test: compile vector_add to AMD ELF and show stats.
pub fn test_amd() {
    let kernel = make_test_kernel();
    println!("=== Compiling vector_add to AMD GCN ELF ===\n");
    match emit_llvm::compile_to_binary(&kernel, GpuTarget::Amdgpu) {
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
pub fn test_complex() {
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
        device_functions: Vec::new(),
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    };

    println!("=== NVIDIA PTX (neuron_activate, O3) ===\n");
    match emit_llvm::compile_to_binary(&kernel, GpuTarget::Nvptx) {
        Ok(ptx) => {
            println!("{}", String::from_utf8_lossy(&ptx));
            println!("=== PTX size: {} bytes ===", ptx.len());
        }
        Err(e) => eprintln!("Error: {}", e),
    }

    println!("\n=== AMD ELF (neuron_activate, O3) ===\n");
    match emit_llvm::compile_to_binary(&kernel, GpuTarget::Amdgpu) {
        Ok(elf) => {
            println!("ELF size: {} bytes", elf.len());
            if elf.len() >= 4 && elf[0..4] == [0x7f, b'E', b'L', b'F'] {
                println!("✓ Valid ELF");
            }
        }
        Err(e) => eprintln!("Error: {}", e),
    }

    println!("\n=== LLVM IR (neuron_activate) ===\n");
    match emit_llvm::compile_to_llvm_ir(&kernel, GpuTarget::Nvptx) {
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
pub fn test_ir() {
    let kernel = make_test_kernel();

    println!("=== NVPTX LLVM IR ===");
    match emit_llvm::compile_to_llvm_ir(&kernel, GpuTarget::Nvptx) {
        Ok(ir) => println!("{}", ir),
        Err(e) => eprintln!("NVPTX error: {}", e),
    }

    println!("\n=== AMDGPU LLVM IR ===");
    match emit_llvm::compile_to_llvm_ir(&kernel, GpuTarget::Amdgpu) {
        Ok(ir) => println!("{}", ir),
        Err(e) => eprintln!("AMDGPU error: {}", e),
    }

    println!("\n=== SPIR-V LLVM IR ===");
    match emit_llvm::compile_to_llvm_ir(&kernel, GpuTarget::Spirv) {
        Ok(ir) => println!("{}", ir),
        Err(e) => eprintln!("SPIR-V error: {}", e),
    }
}

/// Test: compile vector_add to SPIR-V binary.
pub fn test_spirv() {
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
                    println!("Valid SPIR-V binary (magic 0x07230203)");
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
    match emit_llvm::compile_to_llvm_ir(&kernel, GpuTarget::Spirv) {
        Ok(ir) => println!("{}", ir),
        Err(e) => eprintln!("SPIR-V IR error: {}", e),
    }

    // Test shared memory + loop kernel
    println!("\n=== Direct SPIR-V (shared_sum) ===\n");
    let shared_kernel = KernelDef {
        name: "shared_sum".to_string(),
        params: vec![
            KernelParam::FieldRead {
                name: "data".into(),
                slot: 0,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldWrite {
                name: "result".into(),
                slot: 1,
                scalar_type: ScalarType::F32,
            },
        ],
        body: vec![
            KernelOp::SharedDecl {
                id: 0,
                ty: ScalarType::F32,
                count: 64,
            },
            KernelOp::ProtonId { dst: Reg(0) },
            KernelOp::QuarkId { dst: Reg(1) },
            KernelOp::Load {
                dst: Reg(2),
                field: 0,
                index: Reg(1),
                ty: ScalarType::F32,
            },
            KernelOp::SharedStore {
                id: 0,
                index: Reg(0),
                src: Reg(2),
                ty: ScalarType::F32,
            },
            KernelOp::Barrier,
            KernelOp::Const {
                dst: Reg(3),
                value: ConstValue::U32(0),
            },
            KernelOp::Cmp {
                dst: Reg(4),
                a: Reg(0),
                b: Reg(3),
                op: CmpOp::Eq,
                ty: ScalarType::U32,
            },
            KernelOp::Branch {
                cond: Reg(4),
                then_ops: vec![
                    KernelOp::Const {
                        dst: Reg(5),
                        value: ConstValue::F32(0.0),
                    },
                    KernelOp::Const {
                        dst: Reg(6),
                        value: ConstValue::U32(64),
                    },
                    KernelOp::Loop {
                        count: Reg(6),
                        iter_reg: Reg(7),
                        body: vec![
                            KernelOp::SharedLoad {
                                dst: Reg(8),
                                id: 0,
                                index: Reg(7),
                                ty: ScalarType::F32,
                            },
                            KernelOp::BinOp {
                                dst: Reg(5),
                                a: Reg(5),
                                b: Reg(8),
                                op: BinOp::Add,
                                ty: ScalarType::F32,
                            },
                        ],
                    },
                    KernelOp::NucleusId { dst: Reg(9) },
                    KernelOp::Store {
                        field: 1,
                        index: Reg(9),
                        src: Reg(5),
                        ty: ScalarType::F32,
                    },
                ],
                else_ops: vec![],
            },
        ],
        body_source: None,
        next_reg: 10,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    };
    match emit_spirv::emit(&shared_kernel) {
        Ok(spirv) => {
            println!("SPIR-V binary size: {} bytes", spirv.len());
            let path = std::env::temp_dir().join("quanta_shared.spv");
            if std::fs::write(&path, &spirv).is_ok() {
                let val = std::process::Command::new("spirv-val").arg(&path).output();
                match val {
                    Ok(o) if o.status.success() => println!("spirv-val: PASS"),
                    Ok(o) => println!("spirv-val: FAIL\n{}", String::from_utf8_lossy(&o.stderr)),
                    Err(_) => println!("spirv-val not found"),
                }
            }
        }
        Err(e) => eprintln!("Direct SPIR-V error: {}", e),
    }
}

/// Test: compile Rust source via rustc → LLVM IR.
pub fn test_rustc() {
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
    match crate::rustc_compile::rust_to_llvm_ir("vector_add", &params, body) {
        Ok(ir) => {
            println!("{}", ir);
            println!("\n=== Success: {} bytes of LLVM IR ===", ir.len());
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
