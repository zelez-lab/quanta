//! LLVM metadata helpers (NVPTX kernel annotations, etc.).

use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::values::FunctionValue;

pub(crate) fn add_nvptx_kernel_metadata<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    function: &FunctionValue<'ctx>,
) {
    // !nvvm.annotations = !{!0}
    // !0 = !{ptr @kernel_name, !"kernel", i32 1}
    let md_string = context.metadata_string("kernel");
    let md_i32 = context.i32_type().const_int(1, false);
    let fn_val = function.as_global_value().as_pointer_value();

    let md_node = context.metadata_node(&[fn_val.into(), md_string.into(), md_i32.into()]);

    module
        .add_global_metadata("nvvm.annotations", &md_node)
        .unwrap_or(());
}

pub(crate) fn add_spirv_compute_metadata<'ctx>(
    context: &'ctx Context,
    function: &FunctionValue<'ctx>,
) {
    // Vulkan SPIR-V requires "hlsl.shader" attribute set to "compute" on the entry point.
    // This tells LLVM's SPIR-V backend to emit OpEntryPoint GLCompute.
    function.add_attribute(
        inkwell::attributes::AttributeLoc::Function,
        context.create_string_attribute("hlsl.shader", "compute"),
    );
}
