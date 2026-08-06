// Minimal WebGPU type declarations.
//
// Quanta deliberately does NOT depend on `@webgpu/types`. This file
// declares exactly the surface `quanta.ts` (and its helpers) touch —
// about 50 lines of
// hand-written types we own and audit, instead of the ~10K lines that
// ship with the npm package. Under B″ this file will be replaced with
// types generated from the W3C `webgpu.idl`; for B⁰ we hand-author it.
//
// All types are loose: methods accept `any` for descriptor objects.
// This is intentional — the descriptors are built by JS-side `Object`
// literals at the call site, and adding precise types would require
// re-deriving the entire WebIDL hierarchy. The Rust side carries the
// strict typing on the IDL fields.
//
// We use `declare global` so the WebGPU types are available without an
// `import` in every consumer module, matching the way `lib.dom.d.ts`
// surfaces standard browser APIs.
export {};
