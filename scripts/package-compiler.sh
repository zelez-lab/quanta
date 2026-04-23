#!/bin/bash
# Package quanta-compiler for the current platform.
# Usage: ./scripts/package-compiler.sh
#
# Produces: dist/quanta-compiler-{target}.tar.gz

set -euo pipefail

# Detect target triple
case "$(uname -s)-$(uname -m)" in
    Darwin-arm64)  TARGET="aarch64-apple-darwin" ;;
    Darwin-x86_64) TARGET="x86_64-apple-darwin" ;;
    Linux-x86_64)  TARGET="x86_64-unknown-linux-gnu" ;;
    Linux-aarch64) TARGET="aarch64-unknown-linux-gnu" ;;
    *) echo "Unsupported platform: $(uname -s)-$(uname -m)"; exit 1 ;;
esac

VERSION=$(grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')

echo "Building quanta-compiler v${VERSION} for ${TARGET}..."
cargo build -p quanta-compiler --release

mkdir -p dist
ARCHIVE="dist/quanta-compiler-${TARGET}.tar.gz"
tar czf "$ARCHIVE" -C target/release quanta-compiler
shasum -a 256 "$ARCHIVE" > "${ARCHIVE}.sha256"

echo ""
echo "Packaged: ${ARCHIVE}"
echo "SHA256:   $(cat ${ARCHIVE}.sha256)"
echo "Size:     $(du -h ${ARCHIVE} | cut -f1)"
