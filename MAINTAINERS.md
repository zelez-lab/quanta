# Maintainers

Quanta has one maintainer: the repository owner,
[`zelez-lab`](https://github.com/zelez-lab) on GitHub. There is no
committee, no rotation, and no subsystem ownership split — stating that
plainly is more useful than a governance structure the project does not
have.

## How decisions are made

By owner ruling. Design questions, API shape, what lands and what does
not, and when a version is tagged are all decided by the owner. Discussion
is welcome and changes outcomes regularly; it just does not bind the
decision.

Pre-1.0 there is no backward-compatibility promise, so a ruling may
rename or reshape public API. When that happens the change ships with the
migration note that consumers need.

## How to get a change in

Open a pull request — see [CONTRIBUTING.md](CONTRIBUTING.md) for the
`Signed-off-by` requirement, the one-logical-change-per-PR rule, and the
checks to run before pushing. Every merge goes through a PR reviewed by
the owner, including the owner's own work on anything non-trivial.

If a change is large or reshapes an interface, open an issue and agree on
the approach before writing it. A rejected design after the code is
written is a waste of your time, not a verdict on the code.

## Reporting problems

- Bugs and feature requests: <https://github.com/zelez-lab/quanta/issues>
- Security vulnerabilities: see [SECURITY.md](SECURITY.md) — private
  reporting, not a public issue.
- Code of Conduct concerns: see
  [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
