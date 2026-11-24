# The start.sh banner: INKLING in block letters, with MIA AI LAB under it. Sourced by start.sh, which sets M, D and R.
banner() {
  local m=${M:-} d=${D:-} r=${R:-}
  printf '%s' "$m"
  cat <<'ART'
  ██╗███╗   ██╗██╗  ██╗██╗     ██╗███╗   ██╗ ██████╗
  ██║████╗  ██║██║ ██╔╝██║     ██║████╗  ██║██╔════╝
  ██║██╔██╗ ██║█████╔╝ ██║     ██║██╔██╗ ██║██║  ███╗
  ██║██║╚██╗██║██╔═██╗ ██║     ██║██║╚██╗██║██║   ██║
  ██║██║ ╚████║██║  ██╗███████╗██║██║ ╚████║╚██████╔╝
  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝
ART
  printf '%s  %s · TensorFold · RTX PRO 6000 (GPU %s)%s\n' "$d" "${MODEL_LABEL:-Inkling-2-Mini}" "${GPU_ID:-0}" "$r"
  printf '%s  M I A   A I   L A B%s\n' "$m" "$r"
}
