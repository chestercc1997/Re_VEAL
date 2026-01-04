#!/bin/bash

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Source root directory
root_dir="$SCRIPT_DIR/../Benchmarks"
# default_name="dch"  # Use resyn3 directly here
default_name="dc2"  # Use resyn3 directly here

# Traverse all *_dataset_total directories
for dataset_dir in "$root_dir"/*_*_dataset_total; do
    # Ensure it's a directory
    if [[ -d $dataset_dir ]]; then
        echo "Processing directory: $dataset_dir"

        # Create target directories
        mkdir -p "$dataset_dir/${default_name}_S"
        mkdir -p "$dataset_dir/${default_name}_U"

        # Traverse .aig files in this directory
        for file in "$dataset_dir/$default_name"/*.aig; do
            # Ensure file exists
            if [[ -e $file ]]; then
                # Get filename
                filename=$(basename "$file")

                # Use regex to match filename prefix
                if [[ "$filename" =~ ^[0-9]+_[0-9]+_S ]]; then
                    cp "$file" "$dataset_dir/${default_name}_S/"
                    echo "Moved file to ${default_name}_S: $filename"
                elif [[ "$filename" =~ ^[0-9]+_[0-9]+_U ]]; then
                    cp "$file" "$dataset_dir/${default_name}_U/"
                    echo "Moved file to ${default_name}_U: $filename"
                fi
            else
                echo "No .aig files found in: $dataset_dir/$default_name"
            fi
        done
    fi
done
