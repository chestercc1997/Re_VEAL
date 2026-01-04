#!/bin/bash

# ============================================================================
# Script Name: rename_aig_files.sh
# Description: Traverse all .aig files in the specified directory structure and rename them according to rules.
#              Uses GNU Parallel for efficient parallel processing while monitoring runtime and memory usage.
# Requirements:
#      - Install GNU Parallel
#      - Ensure permissions to access all input and output directories
# Usage:
#      chmod +x rename_aig_files.sh
#      ./rename_aig_files.sh
# ============================================================================

# Get script directory and define base directory relative to it
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPT_DIR/../Benchmarks"
MAX_JOBS=64

# Check if base directory exists
if [[ ! -d "$BASE_DIR" ]]; then
    echo "Error: Base directory '$BASE_DIR' does not exist. Please check the path."
    exit 1
fi

# Define function to process .aig files
process_aig_file() {
    local aig_file="$1"
    local dataset_total_dir="$2"

    # Add renaming logic here
    # Example: Generate new filename based on file name or other rules
    local new_filename="${aig_file%.aig}_renamed.aig"  # Example rename
    mv "$aig_file" "$new_filename" && echo "Renamed: $aig_file -> $new_filename" || echo "Failed to rename: $aig_file"
}

export -f process_aig_file  # Export function for use in parallel

# Traverse all *_dataset_total directories and find *_cone_LSB_mfs_U subdirectories, then find .aig files
find "$BASE_DIR" -type d -name "*_dataset_total" | while read -r dataset_total_dir; do
    echo "Entering directory: $dataset_total_dir"

    # Find all *_cone_LSB_mfs_U subdirectories
    find "$dataset_total_dir" -type d -name "*_cone_LSB_mfs_U" | while read -r cone_dir; do
        echo "  Processing subdirectory: $cone_dir"

        # Find all .aig files
        find "$cone_dir" -type f -name "*.aig" | while read -r aig_file; do
            echo "    Found file: $aig_file"

            # Pass .aig file path and corresponding dataset_total_dir to parallel function
            echo "\"$aig_file\" \"$dataset_total_dir\""
        done
    done
done | parallel -j "$MAX_JOBS" --colsep ' ' process_aig_file {1} {2}

echo "All files processed!"
