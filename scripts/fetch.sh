#!/usr/bin/env bash
# Download the upstream Inkling-2-Mini checkpoint (thinkingmachines/Inkling-2-Mini, a pinned revision) into MODEL_DIR,
# or with `drafter` the DFlash2 draft model (incoai/Inkling-2-Mini-DFlash2, a pinned revision) into DRAFTER_DIR.
# Resumable (curl -C -); the weights are checked against the hub's SHA-256 by scripts/prepare.sh.
# Usage: scripts/fetch.sh [DIR]  |  scripts/fetch.sh drafter [DIR]   (HF_TOKEN is sent when set; both repos are public)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh
if [[ "${1:-}" == drafter ]]; then
  DIR="${2:-$DRAFTER_DIR}"
  REPO="$DRAFTER_REPO"
  REV="$DRAFTER_REV"
  FILES=(config.json README.md "$DRAFTER_NAME")
else
  DIR="${1:-$MODEL_DIR}"
  REPO="${HF_REPO:-thinkingmachines/Inkling-2-Mini}"
  REV="${HF_REVISION:-7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6}"
  FILES=(config.json generation_config.json hf_quant_config.json chat_template.jinja processor_config.json
         special_tokens_map.json tokenizer.json tokenizer_config.json tiktoken/tokenizer.model
         model.safetensors.index.json README.md mtp.safetensors model-00001-of-00001.safetensors)
fi
AUTH=(); [[ -n "${HF_TOKEN:-}" ]] && AUTH=(-H "Authorization: Bearer $HF_TOKEN")
mkdir -p "$DIR"
for f in "${FILES[@]}"; do
  mkdir -p "$DIR/$(dirname "$f")"
  log "fetching $f"
  curl -fL --retry 5 --retry-delay 5 -C - "${AUTH[@]}" -o "$DIR/$f" "https://huggingface.co/$REPO/resolve/$REV/$f" \
    || die "could not fetch $f from $REPO@${REV:0:12}"
done
log "$REPO@${REV:0:12} is in $DIR"
