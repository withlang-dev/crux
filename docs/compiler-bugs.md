# Compiler bugs

Issues found in the `with` compiler while bringing Crux up to date.
Compiler version at time of writing: `with v0.13.1-g229a0afaa`.

Spec references are to `~/with/docs/with-specification.md`.

---

## 1. `copy` / `move` operators silently shadow same-named functions

**Severity:** high (silent miscompile, no diagnostic)
**Filed:** withlang-dev/with#295

`copy x` and `move x` are prefix ownership operators (spec §3.7-ish,
ownership section around line 649: `dup(copy xs)`). The parser treats
`copy`/`move` as keywords *unconditionally*, so a user-defined function
named `copy` becomes uncallable: a call like `copy(a, b)` is parsed as
the `copy` operator applied to the tuple expression `(a, b)`, not as a
call to the function.

Defining the function is accepted without warning, and the bad call
compiles to wrong code rather than producing a useful error.

### Repro

```with
fn copy(a: i32, b: i32) -> i32:
    a + b

fn main:
    let x = copy(2, 3)     // parsed as `copy (2, 3)` -> a tuple
    assert(x == 5)
```

```
LLVM verify error
Both operands to ICmp instruction are not of the same type!
  %10 = icmp eq { i32, i32 } %9, i32 5
error: code generation failed
```

`copy(2, 3)` yields a `{ i32, i32 }` tuple (the copy of the tuple
literal) instead of calling `copy`, so the downstream `== 5` compares a
tuple against an `i32`.

### Aggravating factor: no raw-identifier escape

None of the obvious escapes parse, so there is no way to call the
function by its real name:

```
r#copy(2, 3)     // error: expected expression
`copy`(2, 3)     // error: expected expression
@copy(2, 3)      // error: expected expression
```

### Expected

Either (a) reject `fn copy` / `fn move` at definition with a
"reserved word" diagnostic, or (b) resolve `copy(args)` /
`move(args)` as a call when an identifier `copy`/`move` is in scope as
a function (the operator forms are `copy x` / `move x` without
parentheses), or (c) provide a raw-identifier escape.

### Workaround in Crux

`crux.stream.copy` was renamed to `copy_view` (mirroring the existing
`copy_bytes`).

---

## 2. `with <Trait>` clause on a type declaration is not parsed

**Severity:** medium (documented syntax rejected)
**Filed:** withlang-dev/with#296

The spec documents a `with <Trait>, ...` clause for attaching derivable
traits to a type (§2.3 "Copy Types", and §6.1 line ~2178):

```with
type Handle[T] { index: u32, generation: u32 }
    with Copy, Eq, Hash
```

In practice the parser rejects the `with` clause in every placement
tried — including the exact spec example — with
`error: expected declaration (fn, type, enum, let, use, extern)`.

### Repro

All three of these fail at the `with` token:

```with
// (a) inline brace + continuation line (the spec's own form)
type P { x: i32, y: i32 }
    with Copy

// (b) block form + continuation line
type P {
    x: i32,
    y: i32,
}
    with Copy

// (c) block form + same line as closing brace
type P {
    x: i32,
    y: i32,
} with Copy
```

```
error: expected declaration (fn, type, enum, let, use, extern)
 --> w.w:2:5
  |     with Copy
  |     ^^^^
```

### Expected

The `with Copy, ...` clause should parse as shown in the spec.

### Workaround in Crux

Use a free-standing `impl Copy for T` declaration instead, which the
compiler accepts:

```with
type Shape {
    dims: [Size; 8],
    rank: i32,
}
impl Copy for Shape
```
