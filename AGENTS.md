# AGENTS

## Purpose

With exists to build Crux. This repo has a **dual purpose**: implement Crux
as an excellent ML foundation, *and* use the experience of building it to
surface design issues in the With language. The compiler issues we file at
`withlang-dev/with` are a primary deliverable — a clean, idiomatic Crux is
the proving ground that produces them. Treat compiler friction as signal,
not just an obstacle: investigate it, reduce it to a minimal repro, and file
a good issue.

## Working with the With compiler

- Whenever you find a problem with the With compiler, gather the evidence and
  submit an issue to `withlang-dev/with` using the `gh` CLI. Record it in
  `docs/compiler-bugs.md` with a repro and the issue link.
- When filing compiler issues, include references to the relevant sections or
  line ranges in `~/with/docs/with-specification.md` whenever the spec covers
  the expected behavior.
- Distinguish **bugs** (miscompiles, `with check` passes but `with run` fails,
  spec contradicted, silent/unhelpful diagnostics) from **intended-but-arguable
  design** (e.g. operator precedence, an awkward idiom). File the former; raise
  the latter for discussion rather than assuming.
- The With compiler has debug symbols available, so you can attach a debugger
  and inspect crashes directly when gathering evidence.
- LLVM tools are located at `/usr/local/llvm/bin`.

## Building and testing Crux

- Run the full test suite: `bash scripts/test.sh` (runs `with test` over the
  files in `test/`).
- Check a single module without running it: `with check lib/crux/<mod>.w`.
- Compile and run a file: `with run <file>.w`.
- Keep Crux itself clean and idiomatic — it is the demonstration that the
  language works.
