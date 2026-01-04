#!/bin/bash
set -euo pipefail

# This script processes the renamed default_U AIG files into dc2_U and resyn3_U collections.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DATASET_ROOT="$SCRIPT_DIR/../Benchmarks/32_32_dataset_total"
INPUT_DIR="$DATASET_ROOT/default_U"
OUTPUT_DC2="$DATASET_ROOT/dc2_U"
OUTPUT_RESYN3="$DATASET_ROOT/resyn3_U"

ABC_BIN="${ABC_BIN:-${SCRIPT_DIR}/../abc}"
if [[ ! -x "$ABC_BIN" ]]; then
    if [[ -x "/data/abc" ]]; then
        ABC_BIN="/data/abc"
    elif command -v abc >/dev/null 2>&1; then
        ABC_BIN="$(command -v abc)"
    else
        echo "abc binary not found. Set ABC_BIN to its path." >&2
        exit 1
    fi
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Input directory not found: $INPUT_DIR" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DC2" "$OUTPUT_RESYN3"

shopt -s nullglob
AIG_FILES=("$INPUT_DIR"/*.aig)
if [[ ${#AIG_FILES[@]} -eq 0 ]]; then
    echo "No .aig files found in $INPUT_DIR" >&2
    exit 0
fi

for aig_path in "${AIG_FILES[@]}"; do
    filename=$(basename -- "$aig_path")
    stem="${filename%.aig}"

    dc2_stem="${stem/_Multgen_default/_Multgen_dc2}"
    if [[ "$dc2_stem" == "$stem" ]]; then
        dc2_stem="${stem}_dc2"
    fi

    resyn3_stem="${stem/_Multgen_default/_Multgen_resyn3}"
    if [[ "$resyn3_stem" == "$stem" ]]; then
        resyn3_stem="${stem}_resyn3"
    fi

    dc2_out="$OUTPUT_DC2/${dc2_stem}.aig"
    resyn3_out="$OUTPUT_RESYN3/${resyn3_stem}.aig"

    echo "[dc2 ] $filename -> $(basename -- "$dc2_out")"
    "$ABC_BIN" -c "read $aig_path; dc2; strash; write $dc2_out"

    echo "[resyn3] $filename -> $(basename -- "$resyn3_out")"
    "$ABC_BIN" -c "read $aig_path; resyn3; write $resyn3_out"

done
