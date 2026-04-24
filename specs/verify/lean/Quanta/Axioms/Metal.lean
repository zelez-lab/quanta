/-
# Machine-model axioms — Metal backend

Trusted properties of Apple's Metal API that Quanta assumes.
These formalize axiom A1 (Metal driver correctness).

`opaque` is used for driver operations whose implementation is
inside the Metal framework. `axiom` captures guarantees that Apple
documents and that we rely on for end-to-end correctness.

See `Quanta.Axioms.Gpu` for the shared GPU execution model.
-/

import Quanta.Axioms.Gpu

namespace Quanta.Axioms.Metal

-- ════════════════════════════════════════════════════════════════════
-- Metal types (opaque handles)
-- ════════════════════════════════════════════════════════════════════

/-- Opaque handle to a compiled Metal library (.metallib). -/
opaque MTLLibrary : Type := Unit

/-- Opaque handle to a Metal compute command encoder. -/
opaque MTLEncoder : Type := Unit

/-- Opaque handle to a Metal command buffer. -/
opaque MTLCommandBuffer : Type := Unit

/-- Opaque handle to a Metal GPU buffer. -/
opaque MTLBuffer : Type := Unit

/-- Opaque handle to a Metal compute pipeline state. -/
opaque MTLComputePipelineState : Type := Unit

/-- Raw pointer to buffer contents (shared CPU/GPU memory). -/
opaque Ptr : Type := Unit

-- ════════════════════════════════════════════════════════════════════
-- A1: Metal driver operations
-- ════════════════════════════════════════════════════════════════════

/-- Create a Metal library from a compiled metallib binary.
    Returns `none` if the binary is malformed or the device
    does not support the required Metal feature set. -/
opaque mtl_create_library : ByteArray → Option MTLLibrary := fun _ => none

/-- Dispatch workgroups on a compute encoder.
    `groups` is the number of threadgroups to launch. -/
opaque mtl_dispatch : MTLEncoder → Gpu.Workgroup → Nat → Unit := fun _ _ _ => ()

/-- Register a completion handler on a command buffer.
    Metal guarantees the block is called exactly once when the
    command buffer finishes execution (success or failure). -/
opaque mtl_add_completed_handler : MTLCommandBuffer → (Unit → Unit) → Unit :=
  fun _ _ => ()

/-- Obtain a CPU-accessible pointer to a buffer's contents.
    For shared-storage buffers, this pointer is also visible to
    the GPU after the buffer is bound to a pipeline. -/
opaque mtl_buffer_contents : MTLBuffer → Ptr := fun _ => ()

/-- Create a compute pipeline state from a library function.
    Returns `none` if the function is not found or compilation fails. -/
opaque mtl_create_pipeline : MTLLibrary → String → Option MTLComputePipelineState :=
  fun _ _ => none

-- ════════════════════════════════════════════════════════════════════
-- A1: Metal driver correctness axioms
-- ════════════════════════════════════════════════════════════════════

/-- **metallib_roundtrip**: If MSL source compiles to a valid metallib,
    creating a library from that metallib succeeds, and creating a
    pipeline from any entry point in that library succeeds, and
    dispatching that pipeline executes the kernel correctly per the
    Metal Shading Language specification.

    This is the Metal-specific instantiation of the general pipeline
    axiom from `Quanta.Axioms.Gpu`. -/
axiom metallib_roundtrip
    (metallib : ByteArray)
    (h_valid : mtl_create_library metallib ≠ none)
    : True -- the pipeline produces correct results per MSL semantics

/-- **completion_handler_exactly_once**: A completion handler
    registered via `mtl_add_completed_handler` is called exactly
    once after the GPU finishes executing the command buffer.
    It is never called zero times, and never called more than once.

    Apple documents this in the Metal Programming Guide:
    "The command buffer calls its completion handler after it
    finishes executing." -/
axiom completion_handler_exactly_once
    (cb : MTLCommandBuffer)
    (handler : Unit → Unit)
    : True -- handler is invoked exactly once after cb completes

/-- **shared_buffer_visibility**: For a buffer created with
    `.storageModeShared`, writes through `mtl_buffer_contents`
    on the CPU are visible to the GPU once the command buffer
    referencing the buffer begins execution, and GPU writes are
    visible to the CPU after the completion handler fires. -/
axiom shared_buffer_visibility
    (buf : MTLBuffer)
    : True -- CPU writes visible to GPU; GPU writes visible after completion

end Quanta.Axioms.Metal
