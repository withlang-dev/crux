#!/bin/bash
# scripts/emergency_bootstrap.sh
#
# Emergency bootstrap repair script.
#
# Use when: the installed seed compiler (~/.local/bin/with) is too old
# to link stage1 from the current source. This happens when the runtime
# object layout changed (new .o files, renamed symbols, deleted C files)
# but the seed's embedded Link.w still expects the old layout.
#
# What it does: uses the seed to COMPILE stage1 (--emit-obj), then
# links manually with the correct runtime objects, bypassing the
# seed's linker entirely. Once stage1 exists, it has the current
# Link.w and can build stage2 normally.
#
# Usage:
#   bash scripts/emergency_bootstrap.sh
#
# After success:
#   make build
#   make smoke
#   make fixpoint
#   make install    # <- critical: updates seed so this isn't needed again

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# --- Configuration ---
SEED="${WITH:-$(command -v with 2>/dev/null || echo "")}"
if [ -z "$SEED" ]; then
    echo "error: no seed compiler found. Set WITH=/path/to/with or add with to PATH." >&2
    exit 1
fi

LLVM_CLANG="${LLVM_CLANG:-/usr/local/llvm/bin/clang}"
if [ ! -x "$LLVM_CLANG" ]; then
    # Try to find it from the seed's runtime
    SEED_DIR="$(dirname "$SEED")"
    SEED_RUNTIME="$SEED_DIR/runtime"
    if [ -f "$SEED_RUNTIME/llvm_cc" ]; then
        LLVM_CLANG="$(cat "$SEED_RUNTIME/llvm_cc" | tr -d '[:space:]')"
    fi
fi

if [ ! -x "$LLVM_CLANG" ]; then
    echo "error: LLVM clang not found at $LLVM_CLANG" >&2
    echo "       Set LLVM_CLANG=/path/to/clang or ensure /usr/local/llvm/bin/clang exists." >&2
    exit 1
fi

SEED_RUNTIME="$(dirname "$SEED")/runtime"
OUT_LIB="$ROOT/out/lib"
OUT_BIN="$ROOT/out/bin"
OUT_GEN="$ROOT/out/gen"

echo "=== Emergency Bootstrap ==="
echo "  seed:    $SEED"
echo "  clang:   $LLVM_CLANG"
echo "  runtime: $SEED_RUNTIME"
echo ""

# --- Step 1: Create output directories ---
mkdir -p "$OUT_LIB" "$OUT_BIN" "$OUT_GEN"

# --- Step 2: Generate source files ---
echo "[1/8] Generating source files..."

# Version
VERSION="$(bash -c "$(cat "$ROOT/Makefile" | sed -n '/^RESOLVE_VERSION_SH/,/^$/p' | tail -n +2 | head -n -1)" 2>/dev/null || cat "$ROOT/src/version")"
if [ -z "$VERSION" ]; then
    VERSION="$(cat "$ROOT/src/version" | head -1 | tr -d '[:space:]')"
fi
sed "s/WITH_VERSION_PLACEHOLDER/$VERSION/g" "$ROOT/src/main.w" > "$OUT_GEN/main.w"
echo "  main.w (version: $VERSION)"

# Embedded stdlib
if [ -f "$ROOT/scripts/generate_embedded_stdlib.py" ]; then
    python3 "$ROOT/scripts/generate_embedded_stdlib.py" "$ROOT" "$OUT_GEN/embedded_stdlib_runtime.w"
    echo "  embedded_stdlib_runtime.w"
fi

# Compat runtime (concatenation of compat + embedded stdlib)
if [ -f "$ROOT/rt/compat_runtime.w" ] && [ -f "$OUT_GEN/embedded_stdlib_runtime.w" ]; then
    cat "$ROOT/rt/compat_runtime.w" "$OUT_GEN/embedded_stdlib_runtime.w" > "$OUT_GEN/compat_runtime.w"
    echo "  compat_runtime.w"
fi

# --- Step 3: Build C runtime objects ---
echo "[2/8] Building C runtime objects..."

if [ -f "$ROOT/runtime/helpers.c" ]; then
    cc -c "$ROOT/runtime/helpers.c" -o "$OUT_LIB/helpers.o" -I"$ROOT/runtime"
    echo "  helpers.o"
fi

# --- Step 4: Build With runtime objects with seed ---
echo "[3/8] Building With runtime objects..."

build_with_obj() {
    local src="$1"
    local dst="$2"
    local flags="${3:-}"
    if [ -f "$src" ]; then
        "$SEED" build "$src" --emit-obj --no-prelude $flags -o "$dst" 2>/dev/null
        echo "  $(basename "$dst")"
    fi
}

build_with_obj "$OUT_GEN/compat_runtime.w" "$OUT_LIB/compat_runtime.o" "-O0"
build_with_obj "$ROOT/rt/rt_core.w" "$OUT_LIB/rt_core.o" "-O2"
build_with_obj "$ROOT/rt/darwin_aarch64.w" "$OUT_LIB/rt_darwin_aarch64.o" "-O2"
build_with_obj "$ROOT/rt/panic_runtime.w" "$OUT_LIB/panic_runtime.o" "-O0"
build_with_obj "$ROOT/rt/fiber_stubs.w" "$OUT_LIB/fiber_stubs.o" "-O0"
build_with_obj "$ROOT/rt/channel_runtime.w" "$OUT_LIB/channel_runtime.o" "-O0"

# Optional objects (may not exist in all versions)
[ -f "$ROOT/rt/fiber_runtime.w" ] && build_with_obj "$ROOT/rt/fiber_runtime.w" "$OUT_LIB/fiber_runtime.o" "-O0"
[ -f "$OUT_GEN/embedded_stdlib_runtime.w" ] && build_with_obj "$OUT_GEN/embedded_stdlib_runtime.w" "$OUT_LIB/embedded_stdlib_runtime.o" "-O0"

# --- Step 5: Build embedded objects blob ---
echo "[4/8] Embedding runtime objects..."

if [ -f "$ROOT/scripts/embed_runtime_objects.sh" ]; then
    bash "$ROOT/scripts/embed_runtime_objects.sh" "$OUT_LIB" "$OUT_LIB/embedded_objects.s"
    cc -c "$OUT_LIB/embedded_objects.s" -o "$OUT_LIB/embedded_objects.o"
    echo "  embedded_objects.o"
fi

# --- Step 6: Build fiber assembly ---
echo "[5/8] Building fiber assembly..."

if [ -f "$ROOT/runtime/fiber_asm_aarch64.s" ]; then
    cc -c "$ROOT/runtime/fiber_asm_aarch64.s" -o "$OUT_LIB/fiber_asm.o"
    echo "  fiber_asm.o"
fi

# --- Step 7: Compile stage1 object (seed compiles, we link) ---
echo "[6/8] Compiling stage1 object with seed (bypassing seed linker)..."

"$SEED" build "$OUT_GEN/main.w" --emit-obj -O0 -o "$OUT_LIB/stage1.o"
echo "  stage1.o ($(du -h "$OUT_LIB/stage1.o" | cut -f1))"

# --- Step 8: Link stage1 manually ---
echo "[7/8] Linking stage1..."

# Find LLVM bridge objects from seed runtime
LLVM_BRIDGE="$SEED_RUNTIME/llvm_bridge.o"
CLANG_BRIDGE="$SEED_RUNTIME/clang_bridge.o"
LLVM_DYLIB="$SEED_RUNTIME/libwith_llvm_bridge.dylib"

# Read linker response file if it exists
LINK_RSP=""
if [ -f "$SEED_RUNTIME/llvm_link.rsp" ]; then
    LINK_RSP="@$SEED_RUNTIME/llvm_link.rsp"
fi

# Collect all runtime objects
RUNTIME_OBJS=""
for obj in \
    "$OUT_LIB/helpers.o" \
    "$OUT_LIB/compat_runtime.o" \
    "$OUT_LIB/panic_runtime.o" \
    "$OUT_LIB/fiber_stubs.o" \
    "$OUT_LIB/embedded_objects.o" \
    "$OUT_LIB/fiber_asm.o" \
    "$LLVM_BRIDGE" \
    "$CLANG_BRIDGE"; do
    if [ -f "$obj" ]; then
        RUNTIME_OBJS="$RUNTIME_OBJS $obj"
    fi
done

# Link
"$LLVM_CLANG" -fuse-ld=lld \
    "$OUT_LIB/stage1.o" \
    $RUNTIME_OBJS \
    -L"$(dirname "$LLVM_DYLIB")" -lwith_llvm_bridge \
    $LINK_RSP \
    -lSystem -lc++ \
    -o "$OUT_BIN/with-stage1"

echo "  with-stage1 ($(du -h "$OUT_BIN/with-stage1" | cut -f1))"

# Verify stage1 works
if ! "$OUT_BIN/with-stage1" --version >/dev/null 2>&1; then
    echo "error: stage1 binary is not functional" >&2
    exit 1
fi
echo "  verified: $("$OUT_BIN/with-stage1" --version 2>/dev/null)"

# --- Step 9: Build stage2 normally ---
echo "[8/8] Building stage2 with stage1..."

"$OUT_BIN/with-stage1" build "$OUT_GEN/main.w" -o "$OUT_BIN/with-stage2" -O0
cp "$OUT_BIN/with-stage2" "$OUT_BIN/with"

echo ""
echo "=== Emergency Bootstrap Complete ==="
echo ""
echo "  stage1: $OUT_BIN/with-stage1"
echo "  stage2: $OUT_BIN/with-stage2"
echo "  with:   $OUT_BIN/with"
echo ""
echo "Next steps:"
echo "  make build      # validate full stage chain"
echo "  make smoke      # validate selfcheck"
echo "  make fixpoint   # validate stage2 == stage3"
echo "  make install    # update seed so this isn't needed again"