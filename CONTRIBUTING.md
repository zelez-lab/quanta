# Contributing to Quanta

## Developer Certificate of Origin

Every commit must carry a `Signed-off-by` line certifying that you have
the right to submit the contribution under the project's `MIT OR Apache-2.0`
license.

```
Signed-off-by: Your Name <your@email.com>
```

Add it automatically with:

```sh
git commit -s
```

By signing off you certify the following (full text at
<https://developercertificate.org>):

> By making a contribution to this project, I certify that:
>
> (a) The contribution was created in whole or in part by me and I have
>     the right to submit it under the open source license indicated in
>     the file; or
>
> (b) The contribution is based upon previous work that, to the best of
>     my knowledge, is covered under an appropriate open source license
>     and I have the right under that license to submit that work with
>     modifications, whether created in whole or in part by me, under
>     the same open source license; or
>
> (c) The contribution was provided directly to me by some other person
>     who certified (a), (b), or (c) and I have not modified it.
>
> (d) I understand and agree that this project and the contribution are
>     public and that a record of the contribution (including all
>     personal information I submit with it) is maintained indefinitely
>     and may be redistributed consistent with this project or the open
>     source license(s) involved.

## Pull requests

- One logical change per PR.
- All tests must pass (`just test`).
- Run `just hooks` once after cloning to enable the pre-commit hook
  (`fmt --check` + `clippy -D warnings`); git does not pick up
  `.githooks/` on its own.
- Run `just fmt` and `just clippy` before pushing.
- If your change touches a verified component, run the relevant Verus or
  Lean check (see `specs/verify/`).
- For new Tier-A features follow the two-commit verified-track recipe:
  proof foundation first (Lean + Verus), typed API second.

## Verified and unverified zones

`crates/` is the proof-required zone — the shipped library surface:
`quanta-core`, `quanta-ir`, `quanta-compiler`, the wasm-lowering crate,
and the companion / ML crates. A change there that alters semantics the
proofs cover must ship with its Lean, Verus, or Kani update in the same
PR, or an explicit scope note in `docs/verification/scope.md` recording
what the proofs no longer reach. `examples/`, `tests/`, `docs/`,
`bench/` and `web/` are the open zone: no proof obligation, normal
review. If you are unsure which side your change lands on,
`docs/verification/scope.md` is the map — it names, level by level, what
is covered and what stays trusted.
