#!/bin/bash

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARKS_DIR="$SCRIPT_DIR/../Benchmarks"

# ---------- Part A: for each *_dataset_multgen, generate ori_aig + dc2/compress2rs/resyn3/dch/default ----------
process_ori_aig_file() {
    aig_file="$1"

    base_name=$(basename "$aig_file" .aig)

    output_file1="$OUTPUT_DIR1/${base_name}_dc2.aig"
    output_file2="$OUTPUT_DIR2/${base_name}_compress2rs.aig"
    output_file3="$OUTPUT_DIR3/${base_name}_resyn3.aig"
    output_file5="$OUTPUT_DIR5/${base_name}_dch.aig"
    output_file6="$OUTPUT_DIR6/${base_name}_default.aig"

    # Execute operations in abc
    ./abc -c "read $aig_file; dc2; strash; write $output_file1"
    ./abc -c "read $aig_file; compress2rs; strash; write $output_file2"
    ./abc -c "read $aig_file; resyn3; write $output_file3"
    ./abc -c "read $aig_file; dch; strash; write $output_file5"
    ./abc -c "read $aig_file; strash; write $output_file6"

    echo "Processed (ori_aig) $aig_file"
}

export -f process_ori_aig_file
export BENCHMARKS_DIR

# Iterate over all *_dataset_multgen directories
for multgen_dir in "$BENCHMARKS_DIR"/*_*_dataset_multgen; do
    if [[ $multgen_dir =~ $BENCHMARKS_DIR/([0-9]+)_([0-9]+)_dataset_multgen ]]; then
        prefix="${BASH_REMATCH[1]}_${BASH_REMATCH[2]}"

        # Find the corresponding *_dataset directory
        dataset_dir="$BENCHMARKS_DIR/${prefix}_dataset"

        # Define paths
        TOTAL_DIR="$BENCHMARKS_DIR/${prefix}_dataset_total"
        ORI_AIG_DIR="${TOTAL_DIR}/ori_aig"
        OUTPUT_DIR1="${TOTAL_DIR}/dc2"
        OUTPUT_DIR2="${TOTAL_DIR}/compress2rs"
        OUTPUT_DIR3="${TOTAL_DIR}/resyn3"
        OUTPUT_DIR5="${TOTAL_DIR}/dch"
        OUTPUT_DIR6="${TOTAL_DIR}/default"
        OUTPUT_DIR7="${TOTAL_DIR}/total"  # Existing directory in your script (not used below but kept)

        # Create necessary directories
        mkdir -p "$ORI_AIG_DIR" "$OUTPUT_DIR1" "$OUTPUT_DIR2" "$OUTPUT_DIR3" "$OUTPUT_DIR5" "$OUTPUT_DIR6" "$OUTPUT_DIR7"

        # Move/copy .aig files to ori_aig
        find "$multgen_dir/aig" "$dataset_dir/unsigned_aig" -name "*.aig" -exec cp {} "$ORI_AIG_DIR" \;

        export OUTPUT_DIR1 OUTPUT_DIR2 OUTPUT_DIR3 OUTPUT_DIR5 OUTPUT_DIR6

        # Process all .aig files in ori_aig (parallel)
        find "$ORI_AIG_DIR" -name "*.aig" | parallel -j 96 process_ori_aig_file {}
    fi
done


# ---------- Part B: take *_dataset_total/default_U/*.aig and produce *_dataset_total/mfs_U/*_mfs.aig ----------
process_defaultU_file_to_mfsU() {
    aig_file="$1"
    if [[ -f "$aig_file" ]]; then
        # Extract prefix from the file path:
        # .../Benchmarks/<p1>_<p2>_dataset_total/default_U/<whatever>.aig
        if [[ $aig_file =~ $BENCHMARKS_DIR/([0-9]+)_([0-9]+)_dataset_total/default_U/(.*) ]]; then
            prefix="${BASH_REMATCH[1]}_${BASH_REMATCH[2]}"
            OUTPUT_DIR_MFS_U="$BENCHMARKS_DIR/${prefix}_dataset_total/mfs_U"
            mkdir -p "$OUTPUT_DIR_MFS_U"

            base_name=$(basename "$aig_file" .aig)
            # Change output file name to end with _mfs.aig instead of _default.aig
            output_file="$OUTPUT_DIR_MFS_U/${base_name/_default/_mfs}.aig"

            # Execute operations in abc
            if ./abc -c "read $aig_file; strash; logic; mfs2 -W 20; mfs; st; dc2 -l; resub -l -K 16 -N 3 -w 100; logic; mfs2 -W 20; mfs; st; iresyn -l; resyn; resyn2; resyn3; dc2 -l; logic; mfs2 -W 20; mfs; st; dc2 -l; resub -l -K 16 -N 3 -w 100; logic; mfs2 -W 20; mfs; st; iresyn -l; resyn; resyn2; resyn3; dc2 -l; strash; write $output_file"; then
                echo "Processed (default_U -> mfs_U) $aig_file"
            else
                echo "Failed to process (default_U -> mfs_U) $aig_file" >&2
            fi
        fi
    fi
}

export -f process_defaultU_file_to_mfsU
export BENCHMARKS_DIR

# Find all .aig files under default_U and process them in parallel
find "$BENCHMARKS_DIR"/*_*_dataset_total/default_U -name "*.aig" | parallel -j 96 process_defaultU_file_to_mfsU {}