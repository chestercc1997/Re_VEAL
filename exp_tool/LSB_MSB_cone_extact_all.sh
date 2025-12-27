#!/bin/bash

# ==============================================================================
# Script: dataset_total_split_and_cone.sh
#
# Purpose:
#   Generate train/test datasets by processing .aig files from:
#     exp_tool/Benchmark/Benchmarks/<N>_<N>_dataset_total/<op>
#   and writing results to:
#     exp_tool/Benchmark/Benchmarks/<N>_<N>_dataset_{train|test}/<...cone...>/
#
# Usage:
#   ./dataset_total_split_and_cone.sh <lmsb_type> <op> <cut_level> <outputsize> \
#       --train <sizes...> --test <sizes...>
#
# Args:
#   lmsb_type   : MSB or LSB
#   op          : operation name (used as directory under dataset_total)
#   cut_level   : default cut level (may be overridden per file by filename prefix)
#   outputsize  : R parameter for &cone; also used to compute O for MSB
#
# Notes:
#   - All paths are RELATIVE to the repository root:
#       /home/jovyan/workspace/sat_mul_amulet
#   - The effective exp_tool directory is:
#       /home/jovyan/workspace/sat_mul_amulet/exp_tool
#   - This script assumes the 'abc' binary is available at:
#       exp_tool/abc
#     If your abc is elsewhere, adjust ABC_BIN below.
# ==============================================================================

set -u

# ----- Resolve repo root (directory of this script -> repo root) -----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"   # /home/jovyan/workspace/sat_mul_amulet

# ----- Use repo-relative paths (NO /data/...) -----
EXP_TOOL_DIR="$REPO_ROOT/exp_tool"
REV_SCA_DIR="$EXP_TOOL_DIR/Benchmark"
BENCHMARKS_DIR="$REV_SCA_DIR/Benchmarks"

# ----- abc binary (repo-relative) -----
ABC_BIN="$REPO_ROOT/abc"

# ----- Read positional args -----
lmsb_type="$1"
op="$2"
cut_level="$3"
outputsize="$4"
shift 4  # Remove the first 4 positional args

# ----- Parse remaining args into train/test size arrays -----
data_sizes_train=()
data_sizes_test=()
current_section=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train)
            current_section="train"
            ;;
        --test)
            current_section="test"
            ;;
        *)
            if [[ "$current_section" == "train" ]]; then
                data_sizes_train+=("$1")
            elif [[ "$current_section" == "test" ]]; then
                data_sizes_test+=("$1")
            else
                echo "Error: data size '$1' provided before specifying --train or --test." >&2
                exit 1
            fi
            ;;
    esac
    shift
done

if (( ${#data_sizes_train[@]} == 0 && ${#data_sizes_test[@]} == 0 )); then
    echo "Error: No data sizes provided." >&2
    exit 1
fi

# ----- Runtime log (repo-relative) -----
OUTPUT_DIR="$EXP_TOOL_DIR"
output_csv="$OUTPUT_DIR/runtime_log.csv"
mkdir -p "$OUTPUT_DIR"
echo "base_name,runtime_seconds,dataset_type" > "$output_csv"

# Only one op type is provided via CLI, but keep array for future extension
op_types=("$op")

# ----- Sanity checks -----
if [[ ! -d "$BENCHMARKS_DIR" ]]; then
    echo "Error: Benchmarks directory not found: $BENCHMARKS_DIR" >&2
    exit 1
fi

if [[ ! -x "$ABC_BIN" ]]; then
    echo "Error: abc not found or not executable at: $ABC_BIN" >&2
    echo "Tip: set ABC_BIN to your abc path." >&2
    exit 1
fi

# ==============================================================================
# Function: process_files
#   dataset_type: train|test
#   data_size   : e.g., 32, 64, 128...
#   op_type     : operation name
#   lmsb_type   : MSB|LSB
#   cut_level   : default cut level (may be overridden per file)
#   outputsize  : output size for &cone
# ==============================================================================
process_files() {
    local dataset_type="$1"
    local data_size="$2"
    local op_type="$3"
    local lmsb_type="$4"
    local cut_level="$5"
    local outputsize="$6"

    # Input: total dataset directory
    local input_dir="$BENCHMARKS_DIR/${data_size}_${data_size}_dataset_total/${op_type}"

    # Output: split dataset directory
    local output_base="$BENCHMARKS_DIR/${data_size}_${data_size}_dataset_${dataset_type}"
    local output_dir="$output_base/${data_size}_${data_size}_cone_${lmsb_type}_${op_type}"

    if [[ ! -d "$input_dir" ]]; then
        echo "Input directory not found: $input_dir" >&2
        return
    fi

    # Ensure output directory exists and is empty
    mkdir -p "$output_dir"
    if find "$output_dir" -mindepth 1 -print -quit >/dev/null 2>&1; then
        find "$output_dir" -mindepth 1 -delete
    fi

    # Collect .aig files safely
    shopt -s nullglob
    local aig_files=("$input_dir"/*.aig)
    shopt -u nullglob

    if (( ${#aig_files[@]} == 0 )); then
        echo "No AIG files found in: $input_dir"
        return
    fi

    # For MSB mode: O = data_size * 2 - outputsize
    local O_value=$(( data_size * 2 - outputsize ))

    for aig_file in "${aig_files[@]}"; do
        local file_name
        file_name="$(basename "$aig_file")"

        # Optionally override cut_level based on the leading number in the filename
        local resolved_cut_level="$cut_level"
        if [[ "$file_name" =~ ^([0-9]+) ]]; then
            local number="${BASH_REMATCH[1]}"
            echo "Extracted leading number: $number from file: $file_name"

            if (( number == 32 )); then
                resolved_cut_level=15
            elif (( number >= 33 && number <= 64 )); then
                resolved_cut_level=17
            elif (( number == 128 )); then
                resolved_cut_level=19
            elif (( number == 256 )); then
                resolved_cut_level=21
            fi
        fi

        echo "cut_level (resolved): $resolved_cut_level"
        echo "Processing: $aig_file"

        local base_name
        base_name="$(basename "$aig_file" .aig)"
        local output_file="$output_dir/${base_name}.aig"

        local start_time
        start_time="$(date +%s.%N)"

        case "$lmsb_type" in
            MSB)
                "$ABC_BIN" -c "read $aig_file; strash; ps; &get; &cone -c -R $outputsize -O $O_value; &put; topmost -N $resolved_cut_level; ps; write $output_file;"
                ;;
            LSB)
                "$ABC_BIN" -c "read $aig_file; strash; ps; &get; &cone -c -R $outputsize -O 0; &put; ps; write $output_file;"
                # Kept from original script (debug/inspection command)
                "$ABC_BIN" -c "read $aig_file; strash; ps; &get; &cone -c -R $outputsize -O 0; &slice -S 2; &ps"
                ;;
            *)
                echo "Invalid lmsb_type: $lmsb_type (must be MSB or LSB)." >&2
                exit 1
                ;;
        esac

        local end_time
        end_time="$(date +%s.%N)"
        local runtime
        runtime="$(echo "$end_time - $start_time" | bc)"

        echo "Wrote: $output_file"
        echo "$base_name,$runtime,$dataset_type" >> "$output_csv"
    done
}

# ----- Generate train dataset -----
if (( ${#data_sizes_train[@]} > 0 )); then
    for data_size in "${data_sizes_train[@]}"; do
        for op_type in "${op_types[@]}"; do
            process_files "train" "$data_size" "$op_type" "$lmsb_type" "$cut_level" "$outputsize"
        done
    done
else
    echo "No training data sizes supplied; skipping train dataset generation."
fi

# ----- Generate test dataset -----
if (( ${#data_sizes_test[@]} > 0 )); then
    for data_size in "${data_sizes_test[@]}"; do
        for op_type in "${op_types[@]}"; do
            process_files "test" "$data_size" "$op_type" "$lmsb_type" "$cut_level" "$outputsize"
        done
    done
else
    echo "No testing data sizes supplied; skipping test dataset generation."
fi