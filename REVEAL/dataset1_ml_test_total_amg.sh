#!/bin/bash

# ==============================================================================
# Usage:
#   ./dataset1_ml_test_total_amg.sh <lmsb_type> <op> --train <sizes...> --test <sizes...>
#
# Example:
#   ./dataset1_ml_test_total_amg.sh U AM  --train 16 32 64 --test 128 256
#
# This script:
#   1) Parses train/test size lists from CLI arguments.
#   2) For each size, reads .aig files from RevSCA Benchmarks directories.
#   3) Runs abc2 to generate edgelist, node features, and class_map json/csv.
#   4) Extracts i/o, and, lev metrics from abc2 output (fallback to 0 on failure).
#   5) Writes per-graph feature files and per-graph class_map labels.
#   6) Logs runtime to a CSV file.
#
# Notes:
#   - All paths are relative to this script location:
#       /home/jovyan/workspace/sat_mul_amulet/HOGA/dataset1_ml_test_total_amg.sh
#   - RevSCA is assumed at:
#       /home/jovyan/workspace/sat_mul_amulet/exp_tool/Benchmark
#   - abc2 is assumed to be available in the same directory as this script,
#     or adjust ABC2_BIN below.
# ==============================================================================

set -u

# ----- Resolve script directory (absolute), then build relative-based paths -----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"  # /home/jovyan/workspace/sat_mul_amulet

# ----- Binaries (relative to script) -----
ABC2_BIN="$REPO_ROOT/abc2"

# ----- Input arguments -----
lmsb_type="$1"
op="$2"
shift 2  # Remove first two positional args; the rest are selectors and sizes

data_sizes_train=()
data_sizes_test=()
current_section=""

# ----- Parse sizes after --train / --test -----
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

# ----- Output root (relative to repo/script) -----
# Using repo-relative path instead of absolute: /data/HOGA
OUTPUT_ROOT="$SCRIPT_DIR/out"
mkdir -p "$OUTPUT_ROOT"

# Runtime CSV (relative path)
csv_file="$OUTPUT_ROOT/runtime_log.csv"
echo "base_name,runtime_seconds" > "$csv_file"

# ----- Output datasets (relative paths) -----
OUTPUT_DIR_TEST="$SCRIPT_DIR/data_4_ml_test_${op}_${lmsb_type}"
OUTPUT_DIR_TRAIN="$SCRIPT_DIR/data_4_ml_train_${op}_${lmsb_type}"

echo "lmsb_type: $lmsb_type"
echo "op: $op"
echo "data_sizes_train: ${data_sizes_train[*]}"
echo "data_sizes_test: ${data_sizes_test[*]}"
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "REPO_ROOT: $REPO_ROOT"
echo "ABC2_BIN: $ABC2_BIN"

# ----- Clean previous outputs -----
rm -rf "$OUTPUT_DIR_TEST" "$OUTPUT_DIR_TRAIN"

# ----- Create output dirs -----
mkdir -p "$OUTPUT_DIR_TEST" || { echo "Failed to create directory: $OUTPUT_DIR_TEST" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR_TRAIN" || { echo "Failed to create directory: $OUTPUT_DIR_TRAIN" >&2; exit 1; }

echo "Output directory (test):  $OUTPUT_DIR_TEST"
echo "Output directory (train): $OUTPUT_DIR_TRAIN"

# ----- benchmarks root (relative to repo) -----
REV_SCA_ROOT="$REPO_ROOT/exp_tool/Benchmark"
BENCHMARKS_ROOT="$REV_SCA_ROOT/Benchmarks"

# ----- Basic checks -----
if [[ ! -d "$BENCHMARKS_ROOT" ]]; then
    echo "Error: Benchmarks root not found: $BENCHMARKS_ROOT" >&2
    exit 1
fi

if [[ ! -x "$ABC2_BIN" ]]; then
    echo "Error: abc2 not found or not executable at: $ABC2_BIN" >&2
    echo "Tip: put abc2 next to this script, or update ABC2_BIN." >&2
    exit 1
fi

# ==============================================================================
# Helper: parse labels from filename and write class_map.txt
# ==============================================================================
write_class_map_txt() {
    local filename="$1"
    local class_map_file="$2"

    # part1 from filename patterns
    local part1=""
    if [[ "$filename" == *_U_SP* ]] || [[ "$filename" == *_U1* ]] || [[ "$filename" == *_S_SP* ]] || [[ "$filename" == *_S1* ]]; then
        part1="1"
    fi
    if [[ "$filename" == *_U2* ]] || [[ "$filename" == *_S2* ]]; then
        part1="2"
    fi

    # part2 and part3 based on underscore-separated tokens near the end
    local part2
    local part3
    part2="$(echo "$filename" | awk -F'_' '{print $(NF-3)}')"
    part3="$(echo "$filename" | awk -F'_' '{print $(NF-2)}')"

    echo "part1: $part1"
    echo "part2: $part2"
    echo "part3: $part3"

    # Map to numeric codes
    local output_part1="$part1"
    local output_part2=""
    local output_part3=""
    local output_part4=""

    case "$part2" in
        AR)   output_part2="1" ;;
        DT)   output_part2="2" ;;
        WT)   output_part2="3" ;;
        CWT)  output_part2="4" ;;
        4to2) output_part2="5" ;;
        BD)   output_part2="6" ;;
        CN)   output_part2="7" ;;
        CT)   output_part2="8" ;;
        OS)   output_part2="9" ;;
    esac

    case "$part3" in
        RC)  output_part3="1" ;;
        SE)  output_part3="2" ;;
        CL)  output_part3="3" ;;
        CK)  output_part3="4" ;;
        HCA) output_part3="5" ;;
        LF)  output_part3="6" ;;
        KS)  output_part3="7" ;;
        BK)  output_part3="8" ;;
        JCA) output_part3="9" ;;
        CU)  output_part3="10" ;;
        CS)  output_part3="11" ;;
    esac

    case "$part3" in
        RC|SE|CL|CK|CU|CS) output_part4="1" ;;
        KS|LF|HCA|JCA|BK)  output_part4="2" ;;
    esac

    # Write one value per line
    {
        echo "$output_part1"
        echo "$output_part2"
        echo "$output_part3"
        echo "$output_part4"
    } >> "$class_map_file"
}

export -f write_class_map_txt

# ==============================================================================
# Process test set
# ==============================================================================
for data_size_test in "${data_sizes_test[@]}"; do
    TARGET_DIR_TEST="$BENCHMARKS_ROOT/${data_size_test}_${data_size_test}_dataset_test/${data_size_test}_${data_size_test}_cone_${lmsb_type}_${op}"
    echo "Checking directory: $TARGET_DIR_TEST"

    if [[ ! -d "$TARGET_DIR_TEST" ]]; then
        echo "Warning: directory not found, skipping: $TARGET_DIR_TEST" >&2
        continue
    fi

    for file in "$TARGET_DIR_TEST"/*.aig; do
        [[ -e "$file" ]] || continue

        filename="$(basename "$file" .aig)"

        mkdir -p "$OUTPUT_DIR_TEST/$filename/edgelist" \
                 "$OUTPUT_DIR_TEST/$filename/feature" \
                 "$OUTPUT_DIR_TEST/$filename/class_map"

        # Run abc2 and time it
        start_time="$(date +%s.%N)"
        output="$("$ABC2_BIN" -c "read $file; strash; &get;&ps;&edgelist -m -F test.el -c test-class_map.json -f test-feats.csv" 2>&1)"
        status=$?
        end_time="$(date +%s.%N)"
        runtime="$(echo "$end_time - $start_time" | bc)"

        if [[ $status -ne 0 ]]; then
            echo "Warning: abc2 failed for $file (status $status). Defaulting graph metrics to zero." >&2
            io_value=0
            and_value=0
            lev_value=0
        else
            io_value="$(echo "$output" | grep -oP 'i/o = \s*\K\d+' || true)"
            and_value="$(echo "$output" | grep -oP 'and = \s*\K\d+' || true)"
            lev_value="$(echo "$output" | grep -oP 'lev = \s*\K\d+' || true)"

            if [[ -z "${io_value:-}" || -z "${and_value:-}" || -z "${lev_value:-}" ]]; then
                echo "Warning: Missing metrics in abc2 output for $file. Falling back to zeros." >&2
                io_value=0
                and_value=0
                lev_value=0
            fi
        fi

        # Write graph_feature.txt (one value per line)
        graph_feature_file="$OUTPUT_DIR_TEST/$filename/feature/graph_feature.txt"
        {
            printf '%s\n' "$data_size_test"
            printf '%s\n' "$io_value"
            printf '%s\n' "$and_value"
            printf '%s\n' "$lev_value"
        } > "$graph_feature_file"

        # Log runtime
        echo "$filename,$runtime" >> "$csv_file"

        # Copy generated outputs to their directories
        cp -f test-feats.csv "$OUTPUT_DIR_TEST/$filename/feature/"
        cp -f test-class_map.json "$OUTPUT_DIR_TEST/$filename/feature/"
        cp -f test.el "$OUTPUT_DIR_TEST/$filename/edgelist/"
        rm -f "$OUTPUT_DIR_TEST/$filename/feature/test-feats.csv"

        # Build class_map.txt derived from filename
        class_map_file="$OUTPUT_DIR_TEST/$filename/class_map/class_map.txt"
        write_class_map_txt "$filename" "$class_map_file"
    done
done

# ==============================================================================
# Process train set
# ==============================================================================
for data_size_train in "${data_sizes_train[@]}"; do
    TARGET_DIR_TRAIN="$BENCHMARKS_ROOT/${data_size_train}_${data_size_train}_dataset_train/${data_size_train}_${data_size_train}_cone_${lmsb_type}_${op}"
    echo "Checking directory: $TARGET_DIR_TRAIN"

    if [[ ! -d "$TARGET_DIR_TRAIN" ]]; then
        echo "Warning: directory not found, skipping: $TARGET_DIR_TRAIN" >&2
        continue
    fi

    for file in "$TARGET_DIR_TRAIN"/*.aig; do
        [[ -e "$file" ]] || continue

        filename="$(basename "$file" .aig)"

        mkdir -p "$OUTPUT_DIR_TRAIN/$filename/edgelist" \
                 "$OUTPUT_DIR_TRAIN/$filename/feature" \
                 "$OUTPUT_DIR_TRAIN/$filename/class_map"

        # Run abc2
        output="$("$ABC2_BIN" -c "read $file; strash; &get;&ps; &edgelist -m -F train.el -c train-class_map.json -f train-feats.csv" 2>&1)"
        status=$?

        if [[ $status -ne 0 ]]; then
            echo "Warning: abc2 failed for $file (status $status). Defaulting graph metrics to zero." >&2
            io_value=0
            and_value=0
            lev_value=0
        else
            io_value="$(echo "$output" | grep -oP 'i/o = \s*\K\d+' || true)"
            and_value="$(echo "$output" | grep -oP 'and = \s*\K\d+' || true)"
            lev_value="$(echo "$output" | grep -oP 'lev = \s*\K\d+' || true)"

            if [[ -z "${io_value:-}" || -z "${and_value:-}" || -z "${lev_value:-}" ]]; then
                echo "Warning: Missing metrics in abc2 output for $file. Falling back to zeros." >&2
                io_value=0
                and_value=0
                lev_value=0
            fi
        fi

        # Write graph_feature.txt (one value per line)
        graph_feature_file="$OUTPUT_DIR_TRAIN/$filename/feature/graph_feature.txt"
        {
            printf '%s\n' "$data_size_train"
            printf '%s\n' "$io_value"
            printf '%s\n' "$and_value"
            printf '%s\n' "$lev_value"
        } > "$graph_feature_file"

        # Copy generated outputs to their directories
        cp -f train-feats.csv "$OUTPUT_DIR_TRAIN/$filename/feature/"
        cp -f train-class_map.json "$OUTPUT_DIR_TRAIN/$filename/feature/"
        cp -f train.el "$OUTPUT_DIR_TRAIN/$filename/edgelist/"
        rm -f "$OUTPUT_DIR_TRAIN/$filename/feature/train-feats.csv"

        # Build class_map.txt derived from filename
        class_map_file="$OUTPUT_DIR_TRAIN/$filename/class_map/class_map.txt"
        write_class_map_txt "$filename" "$class_map_file"
    done
done