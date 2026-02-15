#!/bin/bash
# Workflow: once uploading is done, ensure data is in /tmp on cluster, then generate reasoning models plot.
# Usage: ./run_reasoning_plot_after_upload.sh [--skip-copy]
# Use --skip-copy if data is already in /tmp/llm_data

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LLM_DATA_SRC="${PROJECT_ROOT}/analysis/llm_analysis/llm_data"
LLM_DATA_TMP="/tmp/llm_data"
OUTPUT_PNG="${SCRIPT_DIR}/reasoning_models_o4_o3_comparison.png"

SKIP_COPY=false
for arg in "$@"; do
    if [ "$arg" = "--skip-copy" ]; then
        SKIP_COPY=true
        break
    fi
done

echo "=== Reasoning models plot workflow ==="

# Step 1: Ensure data is in /tmp on cluster
if [ "$SKIP_COPY" = false ]; then
    if [ -d "$LLM_DATA_SRC" ]; then
        echo "Copying llm_data to /tmp (large folder)..."
        mkdir -p /tmp
        if [ -d "$LLM_DATA_TMP" ]; then
            echo "  /tmp/llm_data already exists, skipping copy (use --skip-copy to always skip)"
        else
            cp -r "$LLM_DATA_SRC" "$LLM_DATA_TMP"
            echo "  Done: $LLM_DATA_TMP"
        fi
    else
        echo "Source llm_data not found at $LLM_DATA_SRC"
        echo "Assuming data is already in $LLM_DATA_TMP (e.g. after upload)"
    fi
else
    echo "Skipping copy (--skip-copy)"
fi

if [ ! -d "$LLM_DATA_TMP" ]; then
    echo "ERROR: No data at $LLM_DATA_TMP. Run without --skip-copy or ensure upload places data there."
    exit 1
fi

# Step 2: Generate reasoning models comparison plot
echo ""
echo "Generating reasoning models plot (o4 mini vs o3 mini low/medium/high)..."
cd "$SCRIPT_DIR"
python3 reasoning_models.py \
    --data-dir "$LLM_DATA_TMP" \
    --output "$OUTPUT_PNG" \
    --no-show

echo ""
echo "Done. Saved: $OUTPUT_PNG"
