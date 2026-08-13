# Security Policy

## Supported versions

Quanta is pre-1.0 and ships as tagged alphas. Only the **latest tagged
alpha** is supported: fixes land on `main` and reach users through the next
tag. Older tags are not patched and no backport happens — if you are behind,
upgrade first and confirm the issue still reproduces.

## Reporting a vulnerability

Report privately through GitHub Security Advisories:

<https://github.com/zelez-lab/quanta/security/advisories/new>

Please do not open a public issue, pull request, or discussion for a
suspected vulnerability — the advisory is private until a fix is out, which
is what gives users a chance to upgrade.

Useful things to include:

- The version or commit you hit it on, and the platform + backend (Metal /
  Vulkan / CPU software / WebGPU).
- A reproducer, ideally a small kernel or example.
- What an attacker gets out of it, if that is not obvious.

Things that are in scope: memory unsafety reachable from safe API use,
sandbox or capability escapes, anything where untrusted input (shader
source, IR, a `.npy`/`.npz` file, a tokenizer model) drives the process
somewhere it should not go.

## What to expect

This is a single-maintainer, pre-1.0 project. Handling is **best-effort**:
there is no response-time commitment and no embargo process beyond keeping
the advisory private until a fix is tagged. You will get an
acknowledgement when the report is read, and the advisory is where the
status lives after that.

If a report turns out to be a plain bug rather than a vulnerability, it
gets moved to a public issue and handled normally.
