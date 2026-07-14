/-
# WebIDL grammar mirror — minimal shape

Step **B″** of the FFI TCB shrink track. Models the slice of WebIDL
that Quanta's `extern "C"` boundary depends on: enum declarations,
dictionary-member shapes, and method signatures. Generated from the
same `weedle`-parsed AST that emits the Rust + TypeScript spec tables
(see `crates/lang/quanta-codegen`), so all three sides come from one parse.

Status (2026-04-28, first B″ commit):
- `EnumDecl` is the load-bearing model; the conformance theorem
  `Quanta.Theorems.IdlConformance.quanta_strings_in_spec` discharges
  the enum-string component of T1707 (the `quanta_abi_faithful`
  axiom) against the generated `Quanta.Idl.WebGpuSpec` data.
- `DictionaryDecl` and `MethodDecl` are stubbed for the future
  signature-conformance pass — declared so reviewers can see where
  the shape will land, but no theorems currently consume them. The
  next B″ commits will fill these in and prove the matching
  dictionary-builder + method-signature obligations.

Why a hand-rolled mini-IDL instead of importing a library: the
Lean ecosystem has no off-the-shelf WebIDL formalization, and the
fragment we need is small. Keeping it project-local matches the
"no transitive deps" policy on the verification side and lets the
generator (`emit_lean.rs`) emit straight Lean literals with no
dependency footprint.
-/

namespace Quanta.Idl

-- ════════════════════════════════════════════════════════════════════
-- Enum declarations
-- ════════════════════════════════════════════════════════════════════

/-- A single WebIDL `enum X { "a", "b", … };` declaration.
    `values` is the source-order list of allowed strings; `name`
    is the IDL identifier (e.g. `"GPUTextureFormat"`). -/
structure EnumDecl where
  name   : String
  values : List String
  deriving Repr, DecidableEq

/-- One WebIDL method parameter, in canonical form. `typeName` is
    the spec's type rendered as a plain string by
    `crates/lang/quanta-codegen` (typedefs preserved verbatim, e.g.
    `"GPUSize64"` stays `"GPUSize64"` rather than resolving to
    `"unsigned long long"`). -/
structure ParamSig where
  typeName : String
  optional : Bool
  deriving Repr, DecidableEq

/-- A single WebIDL method declaration on an interface. Carries
    enough shape for the conformance theorems landed so far:
    - `quanta_methods_in_spec` (T1711) uses `interfaceName` +
      `methodName`.
    - `quanta_call_arities_in_spec` (T1712) uses `requiredArity` +
      `maxArity` + `isVariadic`.
    - `quanta_call_types_in_spec` (T1713) uses `params`. -/
structure MethodSig where
  interfaceName : String
  methodName    : String
  /-- Number of non-optional, non-variadic parameters. -/
  requiredArity : Nat
  /-- Number of declared parameters (optional + required), not
      counting the variadic tail. -/
  maxArity      : Nat
  /-- True iff the last parameter is variadic (`type... ident`). -/
  isVariadic    : Bool
  /-- Parameter shapes in declaration order, excluding the variadic
      tail (gated separately by `isVariadic`). -/
  params        : List ParamSig
  deriving Repr, DecidableEq

/-- The full WebGPU IDL surface Quanta consumes. Populated by
    `crates/lang/quanta-codegen`'s Lean emitter from `web/webgpu.idl`.
    Order of `enums` matches IDL source order. -/
structure WebGpuSpec where
  /-- SHA-256 of `web/webgpu.idl` at codegen time. Stamped here so a
      regeneration that forgets to update one of the three targets
      surfaces as a hash mismatch in code review. -/
  sourceSha256 : String
  enums        : List EnumDecl
  /-- Methods declared on the project-relevant interfaces (`GPU`,
      `GPUDevice`, `GPUBuffer`, …), flattened across `partial
      interface` extensions. -/
  methods      : List MethodSig
  deriving Repr

/-- Look up an enum by name. Returns the first match or `none` if
    the IDL does not declare it. WebIDL forbids same-name redeclaration
    so "first match" is total. -/
def WebGpuSpec.lookupEnum (s : WebGpuSpec) (name : String) : Option EnumDecl :=
  s.enums.find? (fun e => e.name = name)

/-- True iff the spec declares `value` as a member of enum `enumName`. -/
def WebGpuSpec.enumHasValue (s : WebGpuSpec) (enumName value : String) : Bool :=
  match s.lookupEnum enumName with
  | none   => false
  | some e => e.values.contains value

/-- True iff the spec declares a method named `methodName` on
    interface `interfaceName`. Partial interfaces are already
    flattened by the codegen, so this is a single linear scan. -/
def WebGpuSpec.hasMethod (s : WebGpuSpec) (interfaceName methodName : String) : Bool :=
  s.methods.any (fun m => m.interfaceName = interfaceName ∧ m.methodName = methodName)

/-- True iff the spec declares a method on `interfaceName` named
    `methodName` whose declared arity range admits `callArity`:
    `requiredArity ≤ callArity` *and* either `isVariadic = true` or
    `callArity ≤ maxArity`. -/
def WebGpuSpec.callArityValid
    (s : WebGpuSpec) (interfaceName methodName : String) (callArity : Nat) : Bool :=
  s.methods.any (fun m =>
    m.interfaceName = interfaceName
    ∧ m.methodName = methodName
    ∧ m.requiredArity ≤ callArity
    ∧ (m.isVariadic = true ∨ callArity ≤ m.maxArity))

/-- The leading `n` `typeName`s of a method's param list, in order.
    Used by the param-type conformance check so a Quanta call passing
    `n` arguments compares against the first `n` declared params. -/
def MethodSig.leadingTypes (m : MethodSig) (n : Nat) : List String :=
  (m.params.take n).map ParamSig.typeName

/-- True iff the spec declares a method on `interfaceName` named
    `methodName` such that some declared overload's leading param
    type names match `argTypes` exactly *and* the overload's arity
    range admits `argTypes.length`. -/
def WebGpuSpec.callTypesValid
    (s : WebGpuSpec) (interfaceName methodName : String) (argTypes : List String) : Bool :=
  s.methods.any (fun m =>
    m.interfaceName = interfaceName
    ∧ m.methodName = methodName
    ∧ m.requiredArity ≤ argTypes.length
    ∧ (m.isVariadic = true ∨ argTypes.length ≤ m.maxArity)
    ∧ m.leadingTypes argTypes.length = argTypes)

-- ════════════════════════════════════════════════════════════════════
-- Dictionary / method stubs (filled in by future B″ commits)
-- ════════════════════════════════════════════════════════════════════

/-- A WebIDL primitive type sufficient to describe Quanta's FFI
    boundary. Future B″ commits flesh this out (`USVString`, sequences,
    nullable wrappers); the shape is fixed now so dependents compile. -/
inductive IdlType where
  | bool
  | unsignedLong          -- u32
  | unsignedLongLong      -- u64 (passed as f64 across the wasm boundary)
  | doubleTy              -- f64
  | usvString             -- UTF-8 string
  | enumRef     (name : String)
  | dictRef     (name : String)
  | object                -- opaque GPU* handle
  | sequence    (inner : IdlType)
  | nullable    (inner : IdlType)
  deriving Repr, DecidableEq

/-- A single dictionary member: `unsigned long size = 0;` becomes
    `{ name := "size", ty := IdlType.unsignedLong, optional := true,
       defaultLiteral := some "0" }`. -/
structure DictMember where
  name           : String
  ty             : IdlType
  optional       : Bool
  defaultLiteral : Option String
  deriving Repr, DecidableEq

/-- A WebIDL `dictionary GPUFooDescriptor { … };` declaration. -/
structure DictionaryDecl where
  name    : String
  /-- Parent dictionary name (`GPUObjectDescriptorBase` etc.), if any.
      WebIDL inheritance is single-parent, so `Option` suffices. -/
  inherit : Option String
  members : List DictMember
  deriving Repr

/-- A WebIDL method parameter. -/
structure Param where
  name     : String
  ty       : IdlType
  optional : Bool
  deriving Repr, DecidableEq

/-- A WebIDL method signature on an interface (e.g.
    `GPUDevice.createBuffer(GPUBufferDescriptor descriptor)`). -/
structure MethodDecl where
  /-- Owning interface (`"GPUDevice"`). -/
  interfaceName : String
  /-- Method name (`"createBuffer"`). -/
  methodName    : String
  params        : List Param
  /-- Return type. `none` for `undefined` / void. -/
  returnTy      : Option IdlType
  /-- True for `Promise<T>` returns. -/
  isAsync       : Bool
  deriving Repr

end Quanta.Idl
