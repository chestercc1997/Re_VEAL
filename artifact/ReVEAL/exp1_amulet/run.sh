#!/bin/bash

# ============================================================================
# run_dynphaseorderopt.sh
#  .aig 
#      -  GNU Parallel
#      -  dynphaseorderopt  ../exp1/
#      chmod +x run_dynphaseorderopt.sh
#      ./run_dynphaseorderopt.sh
# ============================================================================

set -e
# INPUT_DIR="../128_resyn3/"
INPUT_DIR="../64_resyn3/"
PROCESS_DIR="./"
OUTPUT_DIR="$PROCESS_DIR/64_resyn3_output"

mkdir -p "$OUTPUT_DIR"

cd "$PROCESS_DIR"

# 64
MAX_JOBS=96

MEMORY_LIMIT=$((8 * 1024 * 1024))  # 8GB  KB
TIME_LIMIT=3600                    # 3600 = 1

#  GNU Parallel 
if ! command -v parallel &>/dev/null; then
    echo "GNU parallel  'sudo apt-get install parallel' "
    exit 1
fi

#  .aig 
process_aig() {
    local aig_file="$1"
    local output_dir="$2"

    local base_name
    base_name=$(basename "$aig_file" .aig)

    local log_file="$output_dir/${base_name}.txt"

    local peak_memory=0
    local elapsed_time=0

    #  dynphaseorderopt
    ./amulet -verify -v1 "$aig_file" > "${output_dir}/${base_name}_output.aig" &
    local pid=$!

    local start_time=$(date +%s)

    while kill -0 "$pid" 2>/dev/null; do
        #  RSS KB
        current_memory=$(ps -o rss= -p "$pid" | tr -d ' ')

        if [[ -z "$current_memory" || ! "$current_memory" =~ ^[0-9]+$ ]]; then
            echo " $pid : '$current_memory'" >> "$log_file"
            kill -9 "$pid" 2>/dev/null
            return
        fi

        if (( current_memory > peak_memory )); then
            peak_memory=$current_memory
        fi

        local current_time=$(date +%s)
        elapsed_time=$((current_time - start_time))

        if (( current_memory > MEMORY_LIMIT )); then
            echo " 8GB${current_memory} KB: $pid" >> "$log_file"
            kill -9 "$pid" 2>/dev/null
            return
        fi

        if (( elapsed_time >= TIME_LIMIT )); then
            echo " 36001: $pid (time out)" >> "$log_file"
            kill -9 "$pid" 2>/dev/null
            return
        fi

        sleep 1
    done

    #  dynphaseorderopt 
    wait "$pid"
    local exit_status=$?

    if (( exit_status != 0 )); then
        echo "dynphaseorderopt : $exit_status" >> "$log_file"
    else
        echo "dynphaseorderopt " >> "$log_file"
    fi

    echo "${peak_memory} KB" >> "$log_file"
    echo "${elapsed_time} " >> "$log_file"
}

#  parallel 
export -f process_aig

#  parallel 
export OUTPUT_DIR
export MEMORY_LIMIT
export TIME_LIMIT

#  .aig  GNU Parallel 
find "$INPUT_DIR" -type f -name "*.aig" | parallel -j "$MAX_JOBS" process_aig {} "$OUTPUT_DIR"

echo ""