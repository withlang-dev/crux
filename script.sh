#!/usr/bin/env sh
# Usage: curl -fsSL https://app.factory.ai/cli | sh
set -e

# Check for curl
if ! command -v curl >/dev/null 2>&1; then
    echo "Error: curl is required but not installed." >&2
    exit 1
fi

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

info() { printf '[0;90m%s[0m
' "$*"; }
warn() { printf '[1;33m%s[0m
' "$*"; }
error() { printf '[1;31m%s[0m
' "$*"; exit 1; }

# Detect platform
case "$(uname -s)" in
    Darwin) platform="darwin" ;;
    Linux) platform="linux" ;;
    *) error "Unsupported operating system: $(uname -s). For Windows, use: irm https://yourdomain.com/cli/windows | iex" ;;
esac

# Detect architecture
case "$(uname -m)" in
    x86_64|amd64) architecture="x64" ;;
    arm64|aarch64) architecture="arm64" ;;
    *) error "Unsupported architecture: $(uname -m)" ;;
esac

# Detect AVX2 support for x64 only
# Store the base architecture for ripgrep (which doesn't have baseline versions)
rg_architecture="$architecture"
arch_suffix=""
if [ "$architecture" = "x64" ]; then
  has_avx2=false

  # Linux: check /proc/cpuinfo for avx2 flag (case-insensitive)
  if [ "$platform" = "linux" ]; then
    if grep -q -i avx2 /proc/cpuinfo 2>/dev/null; then
      has_avx2=true
    fi
  fi

  # macOS: check sysctl for AVX2
  if [ "$platform" = "darwin" ]; then
    if sysctl -a 2>/dev/null | grep -q "machdep.cpu.*AVX2"; then
      has_avx2=true
    fi
  fi

  if [ "$has_avx2" = "false" ]; then
    arch_suffix="-baseline"
  fi
fi

droid_architecture="${architecture}${arch_suffix}"

# Set binary name (Unix systems only)
binary_name="droid"
install_name="droid"
rg_binary_name="rg"

VER="0.90.0"
BASE_URL="https://downloads.factory.ai"
URL="$BASE_URL/factory-cli/releases/$VER/$platform/$droid_architecture/$binary_name"
SHA_URL="$BASE_URL/factory-cli/releases/$VER/$platform/$droid_architecture/$binary_name.sha256"
RG_URL="$BASE_URL/ripgrep/$platform/$rg_architecture/$rg_binary_name"
RG_SHA_URL="$BASE_URL/ripgrep/$platform/$rg_architecture/$rg_binary_name.sha256"

BINARY="$TMP/$install_name"
RG_BINARY="$TMP/$rg_binary_name"
info "Downloading Factory CLI v$VER for $platform-$architecture"

curl -fsSL -o "$BINARY" "$URL" || error "Download failed"

info "Fetching and verifying checksum"
SHA="$(curl -fsSL "$SHA_URL")" || error "Failed to fetch checksum"

# Pick the right checksum tool based on platform
if [ "$platform" = "darwin" ]; then
    if command -v shasum >/dev/null 2>&1; then
        ACTUAL="$(shasum -a 256 "$BINARY" | awk '{print $1}')"
    else
        warn "shasum not found, skipping checksum verification"
        ACTUAL=""
    fi
else # Linux
    if command -v sha256sum >/dev/null 2>&1; then
        ACTUAL="$(sha256sum "$BINARY" | awk '{print $1}')"
    elif command -v shasum >/dev/null 2>&1; then
        ACTUAL="$(shasum -a 256 "$BINARY" | awk '{print $1}')"
    else
        warn "No checksum tool found, skipping checksum verification"
        ACTUAL=""
    fi
fi

if [ -n "$ACTUAL" ] && [ -n "$SHA" ]; then
    if [ "$ACTUAL" != "$SHA" ]; then
        error "Checksum verification failed"
    fi
    info "Checksum verification passed"
fi

chmod +x "$BINARY"

# Download and install ripgrep
info "Downloading ripgrep for $platform-$architecture"

curl -fsSL -o "$RG_BINARY" "$RG_URL" || error "Ripgrep download failed"

info "Fetching and verifying ripgrep checksum"
RG_SHA="$(curl -fsSL "$RG_SHA_URL")" || error "Failed to fetch ripgrep checksum"

# Verify ripgrep checksum
if [ "$platform" = "darwin" ]; then
    if command -v shasum >/dev/null 2>&1; then
        RG_ACTUAL="$(shasum -a 256 "$RG_BINARY" | awk '{print $1}')"
    else
        warn "shasum not found, skipping ripgrep checksum verification"
        RG_ACTUAL=""
    fi
else # Linux
    if command -v sha256sum >/dev/null 2>&1; then
        RG_ACTUAL="$(sha256sum "$RG_BINARY" | awk '{print $1}')"
    elif command -v shasum >/dev/null 2>&1; then
        RG_ACTUAL="$(shasum -a 256 "$RG_BINARY" | awk '{print $1}')"
    else
        warn "No checksum tool found, skipping ripgrep checksum verification"
        RG_ACTUAL=""
    fi
fi

if [ -n "$RG_ACTUAL" ] && [ -n "$RG_SHA" ]; then
    if [ "$RG_ACTUAL" != "$RG_SHA" ]; then
        error "Ripgrep checksum verification failed"
    fi
    info "Ripgrep checksum verification passed"
fi

chmod +x "$RG_BINARY"

# Set installation directories
DST="$HOME/.local/bin"
FACTORY_DIR="$HOME/.factory"
FACTORY_BIN_DIR="$FACTORY_DIR/bin"

mkdir -p "$DST" || error "Failed to create $DST directory"
mkdir -p "$FACTORY_BIN_DIR" || error "Failed to create $FACTORY_BIN_DIR directory"

# Stop any running droid processes
if command -v pkill >/dev/null 2>&1; then
    # Use pkill to stop all processes named exactly "droid"
    if pkill -KILL -x "droid" 2>/dev/null; then
        sleep 1
        info "Stopped old droid process(es)"
    fi
fi

# Install droid binary
cp "$BINARY" "$DST/$install_name" || error "Failed to install binary"

# Install ripgrep to factory bin directory
cp "$RG_BINARY" "$FACTORY_BIN_DIR/$rg_binary_name" || error "Failed to install ripgrep"

info "Factory CLI v$VER installed successfully to $DST/$install_name"
info "Ripgrep installed successfully to $FACTORY_BIN_DIR/$rg_binary_name"

# PATH configuration
info "Checking PATH configuration..."

# Check if already in PATH
if echo "$PATH" | grep -q "$DST"; then
    info "PATH already configured"
    warn "Run 'droid' to get started!"
else
    # Detect user's shell from $SHELL environment variable
    case "${SHELL##*/}" in
        zsh)
            SHELL_RC="~/.zshrc"
            ;;
        bash)
            SHELL_RC="~/.bashrc"
            ;;
        *)
            # Fallback for other shells (sh, dash, etc)
            SHELL_RC="~/.profile"
            ;;
    esac

    info "PATH configuration required"
    warn "Add $DST to your PATH:"
    printf "  echo 'export PATH="$DST:\$PATH"' >> $SHELL_RC\n"
    printf "  source $SHELL_RC\n"
    warn "Then run 'droid' to get started!"
fi
