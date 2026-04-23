# Property: Layout Consistency

> **ID:** T6 (push constants), T7 (vertex descriptors), T8 (varyings)
> **Layer:** Compiler + Driver
> **Status:** Specification complete, Lean proof for T8

## T6: Push constant layout

CPU and GPU agree on the byte layout of push constant data.

```
∀ param at slot S with type T:
  CPU offset  = S × 16
  GPU offset  = S × 16 (SPIR-V OpMemberDecorate Offset)
  sizeof(T)   ≤ 16
  → CPU write and GPU read access the same bytes
```

## T7: Vertex descriptor consistency

The Metal MTLVertexDescriptor and Vulkan VkVertexInputAttributeDescription
match the VertexLayout specified by the user.

```
∀ layout in PipelineDesc.vertex_layouts:
  ∀ attr in layout.attributes:
    Metal:  attribute[attr.location].format = attr_format_to_metal(attr.format)
    Vulkan: attribute[attr.location].format = attr_format_to_vulkan(attr.format)
    Both:   attribute[attr.location].offset = attr.offset
```

## T8: Varying location coordination

Vertex output locations match fragment input locations.

```
∀ vertex with N attribute params (param[0] = position):
  vertex output Location[k] = k  for k in 0..N-1 (skip position)
  fragment input Location[k] = k  for k in 0..N-1
  → vertex output[k] = fragment input[k]
```

## Proof

- **Lean:** `specs/proofs/lean/Quanta/VaryingCoord.lean` — `varying_coordination`,
  `position_not_forwarded`
- **Kani:** `specs/proofs/kani/wire_roundtrip.rs` — `push_constant_offset_fits`
- **Tests:** `tests/render_triangle_test.rs` — textured quad verifies UV varyings
