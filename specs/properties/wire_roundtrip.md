# Property: Wire Format Roundtrip

> **ID:** T3
> **Layer:** IR (quanta-ir/src/wire/)
> **Status:** Specification complete, Kani harness written

## Statement

Serializing a KernelDef, CompilerOutput, or ShaderDef and deserializing
the result produces the original value. The encoding is injective.

## Formal

```
∀ x: KernelDef:   deserialize(serialize(x)) = Ok(x)
∀ x: CompilerOutput: deserialize(serialize(x)) = Ok(x)
∀ x: ShaderDef:   deserialize(serialize(x)) = Ok(x)
∀ x: ShaderOutput: deserialize(serialize(x)) = Ok(x)
```

Injectivity:
```
∀ x y: serialize(x) = serialize(y) → x = y
```

## Proof

- **Lean:** `specs/proofs/lean/Quanta/WireFormat.lean` — tag uniqueness, tag totality
- **Kani:** `specs/proofs/kani/wire_roundtrip.rs` — BinOp/ScalarType tag roundtrip,
  KernelOp tag sequentiality, push constant offset bounds
- **Tests:** `tests/host_wire.rs` — integration roundtrip tests with concrete values
