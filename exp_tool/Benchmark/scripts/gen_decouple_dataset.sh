#!/bin/bash

# Set maximum parallel tasks
MAX_JOBS=96

echo "Current working directory: $(pwd)"

# Create symbolic links (uncomment and set paths as needed)
# ln -s /data/cchen/yosys/yosys /data/cchen/genmul/build/bin/yosys
# ln -s /data/cchen/abc/abc /data/cchen/genmul/build/bin/abc

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARKS_DIR="$SCRIPT_DIR/../Benchmarks"

# Define dataset sizes
dataset_sizes=()
#
# dataset_sizes+=("10")
for ((i=32; i<=64; i++)); do
    dataset_sizes+=("$i")
done
dataset_sizes+=("128" "256")

# Define function to process .v files
process_v_file() {
    local v_file="$1"
    local output_dir1="$2"
    local output_dir2="$3"

    local base_name
    base_name=$(basename "$v_file" .v)
    local blif_file="$output_dir1/${base_name}.blif"
    local aig_file="$output_dir2/${base_name}.aig"

    # if [[ -f "$blif_file" && -f "$aig_file" ]]; then
    #     echo "Skipping $v_file: $blif_file and $aig_file already exist."
    #     return
    # fi

    echo "Processing file with yosys: $v_file"
    # Convert Verilog to BLIF using yosys
    # ./yosys -p "hierarchy -auto-top -check; flatten; techmap" -o "$blif_file" "$v_file"

    echo "Converting BLIF to AIG using abc: $blif_file"
    # Convert BLIF to AIG using abc
    ./abc -c "read_blif $blif_file; strash; write $aig_file"
}

# Define function to process .sv files
process_sv_file() {
    local sv_file="$1"
    local output_dir1_multgen="$2"
    local output_dir2_multgen="$3"

    local base_name
    base_name=$(basename "$sv_file" .sv)
    local blif_file="$output_dir1_multgen/${base_name}.blif"
    local aig_file="$output_dir2_multgen/${base_name}.aig"

    # Delete the first 'endmodule' and all content before it
    sed '1,/endmodule/d' "$sv_file" > "${sv_file}.tmp" && mv "${sv_file}.tmp" "$sv_file"

    echo "Processing file with yosys: $sv_file"
    # Convert Verilog to BLIF using yosys
    ./yosys -p "hierarchy -auto-top -check; flatten; techmap" -o "$blif_file" "$sv_file"

    echo "Converting BLIF to AIG using abc: $blif_file"
    # Convert BLIF to AIG using abc
    ./abc -c "read_blif $blif_file; strash; write $aig_file"
}

# Export functions for parallel use
export -f process_v_file
export -f process_sv_file

# Iterate through each dataset_size
for dataset_size in "${dataset_sizes[@]}"; do
    # Define working and output directories
    WORK_DIR="$BENCHMARKS_DIR/${dataset_size}_${dataset_size}_dataset/unsigned_verilog"
    OUTPUT_DIR1="$BENCHMARKS_DIR/${dataset_size}_${dataset_size}_dataset/unsigned_blif"
    OUTPUT_DIR2="$BENCHMARKS_DIR/${dataset_size}_${dataset_size}_dataset/unsigned_aig"

    echo "Processing dataset size: ${dataset_size}x${dataset_size}"
    echo "Working directory: $WORK_DIR"

    # Create output directories (if they don't exist)
    mkdir -p "$WORK_DIR"
    mkdir -p "$OUTPUT_DIR1"
    mkdir -p "$OUTPUT_DIR2"

    # Process .v files
    find "$WORK_DIR" -maxdepth 1 -type f -name "*.v" | parallel -j"$MAX_JOBS" process_v_file {} "$OUTPUT_DIR1" "$OUTPUT_DIR2"

    # Define working and output directories for multgen structure
    WORK_DIR_MULTGEN="$BENCHMARKS_DIR/${dataset_size}_${dataset_size}_dataset_multgen/verilog"
    OUTPUT_DIR1_MULTGEN="$BENCHMARKS_DIR/${dataset_size}_${dataset_size}_dataset_multgen/blif"
    OUTPUT_DIR2_MULTGEN="$BENCHMARKS_DIR/${dataset_size}_${dataset_size}_dataset_multgen/aig"

    # Create output directories (if they don't exist)
    mkdir -p "$WORK_DIR_MULTGEN"
    mkdir -p "$OUTPUT_DIR1_MULTGEN"
    mkdir -p "$OUTPUT_DIR2_MULTGEN"

    # Use GNU parallel to process .sv files in parallel
    find "$WORK_DIR_MULTGEN" -maxdepth 1 -type f -name "*.sv" | parallel -j"$MAX_JOBS" process_sv_file {} "$OUTPUT_DIR1_MULTGEN" "$OUTPUT_DIR2_MULTGEN"
done

echo "All tasks completed."
