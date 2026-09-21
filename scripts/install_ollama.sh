#!/usr/bin/env bash
# Install / refresh a user-local Ollama binary under .tools/
set -euo pipefail

CLOSE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="${CLOSE_ROOT}/.tools"
VERSION="${OLLAMA_VERSION:-v0.32.1}"
URL="${OLLAMA_URL:-https://github.com/ollama/ollama/releases/download/${VERSION}/ollama-linux-amd64.tar.zst}"
TMP="${TMPDIR:-/tmp}/ollama-linux-amd64.$$.tar.zst"

mkdir -p "$TOOLS"
echo "Downloading ${URL}"
curl -fL --progress-bar -o "$TMP" "$URL"
echo "Extracting into ${TOOLS}"
zstd -d -c "$TMP" | tar -C "$TOOLS" -x
rm -f "$TMP"
chmod +x "${TOOLS}/bin/ollama"
"${TOOLS}/bin/ollama" --version || true
echo "Installed: ${TOOLS}/bin/ollama"
