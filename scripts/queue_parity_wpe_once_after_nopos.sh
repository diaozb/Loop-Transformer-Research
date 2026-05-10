#!/usr/bin/env bash
set -euo pipefail

CURRENT_PID="${1:?usage: queue_parity_wpe_once_after_nopos.sh <current_nopos_pid>}"

REPO_ROOT="/data/yizhou/looped-tf-length-generalization"
EXPECTED_CMD="scripts/run_train_ponder_nopos.py --task parity"
QUEUE_LOG="${REPO_ROOT}/logs/queue_parity_wpe_once_after_nopos.log"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "$(dirname "$QUEUE_LOG")"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$1" >> "$QUEUE_LOG"
}

log "Queue watcher started. Waiting on PID ${CURRENT_PID} for current parity+NoPE+Ponder run."

while true; do
  if [[ -r "/proc/${CURRENT_PID}/cmdline" ]]; then
    CMDLINE="$(tr '\0' ' ' < "/proc/${CURRENT_PID}/cmdline")"
    if [[ "$CMDLINE" == *"$EXPECTED_CMD"* ]]; then
      sleep "$POLL_SECONDS"
      continue
    fi
  fi
  break
done

log "Detected current NoPE run has finished. Launching parity+WPE-once+Ponder with n_steps=20."

source /root/miniconda3/etc/profile.d/conda.sh
conda activate ltf
cd "$REPO_ROOT"

if python scripts/run_train_ponder_wpe_once.py --task parity --n-steps 20 >> "$QUEUE_LOG" 2>&1; then
  log "Queued parity+WPE-once+Ponder run finished successfully."
else
  STATUS=$?
  log "Queued parity+WPE-once+Ponder run failed with exit code ${STATUS}."
  exit "$STATUS"
fi
