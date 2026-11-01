#!/usr/bin/env bash
# Fetch the public dataset (not redistributed here) and verify it against data/checksums.sha256.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST="${FAULT_DATA_DIR:-data/pubrepo}"
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/KingArthur000/Electrical-Fault-detection-and-classification "$DEST"
fi
git -C "$DEST" checkout -q 4e094cf
(cd "$DEST" && sha256sum -c "$OLDPWD/data/checksums.sha256")
echo "Dataset ready in $DEST"
