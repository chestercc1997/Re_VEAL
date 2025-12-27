#!/bin/bash

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARKS_DIR="$SCRIPT_DIR/../Benchmarks"

# Traverse all *_dataset_multgen directories
for multgen_dir in "$BENCHMARKS_DIR"/*_*_dataset_multgen; do
    if [[ $multgen_dir =~ $BENCHMARKS_DIR/([0-9]+)_([0-9]+)_dataset_multgen ]]; then
        prefix="${BASH_REMATCH[1]}_${BASH_REMATCH[2]}"

        # Find corresponding *_dataset directory
        dataset_dir="$BENCHMARKS_DIR/${prefix}_dataset"

        # Define paths
        TOTAL_DIR="$BENCHMARKS_DIR/${prefix}_dataset_total"
        ORI_AIG_DIR="${TOTAL_DIR}/ori_aig"
        OUTPUT_DIR1="${TOTAL_DIR}/dc2"
        OUTPUT_DIR2="${TOTAL_DIR}/compress2rs"
        OUTPUT_DIR3="${TOTAL_DIR}/resyn3"
        OUTPUT_DIR5="${TOTAL_DIR}/dch"
        OUTPUT_DIR6="${TOTAL_DIR}/default"
        OUTPUT_DIR7="${TOTAL_DIR}/total"  # New output directory

        # You can add other required operations here, such as creating directories or processing files
        echo "Processing prefix: $prefix"
        echo "Source dataset directory: $dataset_dir"
        echo "Output directory 1: $OUTPUT_DIR1"
        echo "Output directory 2: $OUTPUT_DIR2"
        echo "Output directory 3: $OUTPUT_DIR3"
        echo "Output directory 5: $OUTPUT_DIR5"
        echo "Output directory 6: $OUTPUT_DIR6"
        echo "Total output directory: $OUTPUT_DIR7"

mkdir -p "$OUTPUT_DIR7"

# Copy files to target directory
cp "$OUTPUT_DIR1"/*.aig "$OUTPUT_DIR7/" && echo "Copied files from $OUTPUT_DIR1 to $OUTPUT_DIR7"
cp "$OUTPUT_DIR2"/*.aig "$OUTPUT_DIR7/" && echo "Copied files from $OUTPUT_DIR2 to $OUTPUT_DIR7"
cp "$OUTPUT_DIR3"/*.aig "$OUTPUT_DIR7/" && echo "Copied files from $OUTPUT_DIR3 to $OUTPUT_DIR7"
cp "$OUTPUT_DIR5"/*.aig "$OUTPUT_DIR7/" && echo "Copied files from $OUTPUT_DIR5 to $OUTPUT_DIR7"
cp "$OUTPUT_DIR6"/*.aig "$OUTPUT_DIR7/" && echo "Copied files from $OUTPUT_DIR6 to $OUTPUT_DIR7"

echo "All files copied to $OUTPUT_DIR7"
    fi
done
