#!/bin/bash
# =============================================================================
# SYNC SCRIPT — Move RunPod /workspace to DigitalOcean
# Usage: ./pull_from_runpod.sh
# =============================================================================

# Config
SOURCE="runpod-a100:/workspace/"
DEST="/home/jeanclawdai/runpod-backup"

echo "============================================="
echo "SGLang-Mamba: RunPod -> DO Transfer"
echo "============================================="
echo "  Source:      $SOURCE"
echo "  Destination: $DEST"
echo ""

# Ensure destination exists
mkdir -p "$DEST"

# Start rsync
# -a: Archive mode (preserves permissions, symlinks, etc)
# -z: Compress (faster over internet)
# -P: Show partial progress
# --info=progress2: Single-line global progress bar (cleaner than per-file)
# --stats: Final summary
echo "[1/1] Synchronizing files..."
rsync -azP --info=progress2 --stats "$SOURCE" "$DEST"

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "============================================="
    echo "TRANSFER COMPLETE! ✅"
    echo "============================================="
    echo "  Total size: $(du -sh $DEST | cut -f1)"
else
    echo "============================================="
    echo "TRANSFER FAILED ❌ (Exit code: $EXIT_CODE)"
    echo "============================================="
fi
