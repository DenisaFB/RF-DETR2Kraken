#!/bin/bash
set -e

INPUT=""
OUTPUT=""
CONFIG=""
HTR=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input) INPUT="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --htr) HTR="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

if [ -z "$INPUT" ] || [ -z "$OUTPUT" ] || [ -z "$CONFIG" ]; then
    echo "Usage: ./run_pipeline.sh --input <dir> --output <dir> --config <file> [--htr <model>]"
    exit 1
fi

./setup_envs.sh

echo "=== Running layout stage ==="
conda run -n layout_env python run_layout.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --config "$CONFIG"

echo "=== Running Kraken stage ==="
if [ -n "$HTR" ]; then
    conda run -n kraken_env python run_kraken.py \
        --input "$INPUT" \
        --output "$OUTPUT" \
        --htr "$HTR"
else
    conda run -n kraken_env python run_kraken.py \
        --input "$INPUT" \
        --output "$OUTPUT"
fi

echo "Pipeline finished."