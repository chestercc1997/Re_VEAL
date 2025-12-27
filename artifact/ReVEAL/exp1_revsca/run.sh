# #!/bin/bash

# # ============================================================================
# # run_revscaorderopt.sh
# #  .aig 
# #      
# # 
# #      -  GNU Parallel
# #      -  revscaorderopt  ../exp1/
# # 
# #      chmod +x run_revscaorderopt.sh
# #      ./run_revscaorderopt.sh
# # ============================================================================

# # 
# set -e

# # 
# # INPUT_DIR="../64_resyn3/"
# # INPUT_DIR="../64_compress2rs/"
# # INPUT_DIR="../64_mfs/"
# # INPUT_DIR="../64_dch/"
# INPUT_DIR="../64_dc2/"
# PROCESS_DIR="../exp2/"
# OUTPUT_DIR="$PROCESS_DIR/dc2_64_output"
# # OUTPUT_DIR="$PROCESS_DIR/mfs"
# # OUTPUT_DIR="$PROCESS_DIR/resyn3"
# # OUTPUT_DIR="$PROCESS_DIR/compress2rs"

# # 
# mkdir -p "$OUTPUT_DIR"

# # 
# cd "$PROCESS_DIR"

# # 64
# MAX_JOBS=64

# # 
# MEMORY_LIMIT=$((8 * 1024 * 1024))  # 8GB  KB
# TIME_LIMIT=3600                    # 3600 = 1

# #  GNU Parallel 
# if ! command -v parallel &>/dev/null; then
#     echo "GNU parallel  'sudo apt-get install parallel' "
#     exit 1
# fi

# #  .aig 
# process_aig() {
#     local aig_file="$1"
#     local output_dir="$2"

#     # 
#     local base_name
#     base_name=$(basename "$aig_file" .aig)

#     # 
#     local log_file="$output_dir/${base_name}.txt"

#     # 
#     local peak_memory=0
#     local elapsed_time=0
#     output_aig_file="${output_dir}/${base_name}_output.aig"
#     #  revscaorderopt
#     ./revsca "$aig_file" "$output_aig_file" -u  &
#     local pid=$!

#     # 
#     local start_time=$(date +%s)

#     # 
#     while kill -0 "$pid" 2>/dev/null; do
#         #  RSS KB
#         current_memory=$(ps -o rss= -p "$pid" | tr -d ' ')

#         # 
#         if [[ -z "$current_memory" || ! "$current_memory" =~ ^[0-9]+$ ]]; then
#             echo " $pid : '$current_memory'" >> "$log_file"
#             kill -9 "$pid" 2>/dev/null
#             return
#         fi

#         # 
#         if (( current_memory > peak_memory )); then
#             peak_memory=$current_memory
#         fi

#         # 
#         local current_time=$(date +%s)
#         elapsed_time=$((current_time - start_time))

#         # 
#         if (( current_memory > MEMORY_LIMIT )); then
#             echo " 8GB${current_memory} KB: $pid" >> "$log_file"
#             kill -9 "$pid" 2>/dev/null
#             return
#         fi

#         # 
#         if (( elapsed_time >= TIME_LIMIT )); then
#             echo " 36001: $pid (time out)" >> "$log_file"
#             kill -9 "$pid" 2>/dev/null
#             return
#         fi

#         # 
#         sleep 1
#     done

#     #  revscaorderopt 
#     wait "$pid"
#     local exit_status=$?

#     # 
#     if (( exit_status != 0 )); then
#         echo "revscaorderopt : $exit_status" >> "$log_file"
#     else
#         echo "revscaorderopt " >> "$log_file"
#     fi

#     # 
#     echo "${peak_memory} KB" >> "$log_file"
#     echo "${elapsed_time} " >> "$log_file"
# }

# #  parallel 
# export -f process_aig

# #  parallel 
# export OUTPUT_DIR
# export MEMORY_LIMIT
# export TIME_LIMIT

# #  .aig  GNU Parallel 
# find "$INPUT_DIR" -type f -name "*.aig" | parallel -j "$MAX_JOBS" process_aig {} "$OUTPUT_DIR"

# echo ""
#!/bin/bash

# ============================================================================
# run_revscaorderopt.sh
#  .aig 
#      -  GNU Parallel
#      -  revscaorderopt  ../exp1/
#      chmod +x run_revscaorderopt.sh
#      ./run_revscaorderopt.sh
# ============================================================================

set -e

# 64
MAX_JOBS=64

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
    output_aig_file="${output_dir}/${base_name}_output.aig"

    #  revscaorderopt
    ./revsca "$aig_file" "$output_aig_file" -u &
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

    #  revscaorderopt 
    wait "$pid"
    local exit_status=$?

    if (( exit_status != 0 )); then
        echo "revscaorderopt : $exit_status" >> "$log_file"
    else
        echo "revscaorderopt " >> "$log_file"
    fi

    echo "${peak_memory} KB" >> "$log_file"
    echo "${elapsed_time} " >> "$log_file"
}

#  parallel 
export -f process_aig

#  parallel 
export MEMORY_LIMIT
export TIME_LIMIT

declare -a dirs=(
    "../256_dc2/:../../ReVEAL/exp1_revsca/dc2_256_output"
    "../256_resyn3/:../../ReVEAL/exp1_revsca/resyn3_256_output"
    "../128_dc2/:../../ReVEAL/exp1_revsca/dc2_128_output"
    "../128_resyn3/:../../ReVEAL/exp1_revsca/resyn3_128_output"
)

for dir in "${dirs[@]}"; do
    IFS=':' read -r INPUT_DIR OUTPUT_DIR <<< "$dir"

    mkdir -p "$OUTPUT_DIR"

    #  .aig  GNU Parallel 
    find "$INPUT_DIR" -type f -name "*.aig" | parallel -j "$MAX_JOBS" process_aig {} "$OUTPUT_DIR"
done

echo ""