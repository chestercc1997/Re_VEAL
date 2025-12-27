#!/bin/bash

# Exit immediately if any command exits with a non-zero status
set -e
# Treat unset variables as an error
set -u
# Fail on pipeline errors
set -o pipefail

# Trap for graceful exit on interruption
trap 'echo "Script interrupted."; exit 1;' SIGINT SIGTERM

# Define input and output paths
input_file="../our/pred_stage3_dc2_U_128.csv"
# input_file="../our/pred_test.csv"
dc2_dir="../128_dc2"
default_dir="../128_default"
output_dir1="./cnf_dc2_128"
output_dir2="./dc2_128_result_newp"

# Create output directories
mkdir -p "$output_dir1" "$output_dir2"

# Determine the number of parallel jobs
num_jobs=9

# Function to process a single CSV row
process_row() {
    local col1="$1"
    local col2="$2"
    local col3="$3"
    local col4="$4"

    # Construct .aig input file paths
    local aig_file="$dc2_dir/$col1.aig"
    local aig_file1="$default_dir/$col2.aig"
    local aig_file2="$default_dir/$col3.aig"
    local aig_file3="$default_dir/$col4.aig"

    # Create output subdirectories
    local current_output_dir="$output_dir1/$col1"
    local output_sub_dir="$output_dir2/$col1"
    mkdir -p "$current_output_dir" "$output_sub_dir"

    # Define log file
    local output_log="$output_sub_dir/output.txt"
    if [[ -f "$output_log" && -s "$output_log" ]]; then
        echo ": $output_log"
        return
    fi
    # Clear the log file before writing
    : > "$output_log"

    # Initialize the log file with creation message
    echo ": $output_log" | tee -a "$output_log"

    # Define output CNF files
    local output_file1="$current_output_dir/cnf1.cnf"
    local output_file2="$current_output_dir/cnf2.cnf"
    local output_file3="$current_output_dir/cnf3.cnf"

    # Initialize commands array
    local commands=()
    local temp_files=()

    # Add memory monitoring to each command
    add_command() {
        local cmd="$1"
        local temp_file=$(mktemp)
        temp_files+=("$temp_file")
        commands+=("/usr/bin/time -v $cmd 2> $temp_file")
    }

    #  output_file1
    if [[ -f "$output_file1" ]]; then
        add_command "../../../exp_tool/kissat/kissat-4.0.1-linux-amd64 --unsat \"$output_file1\""
    else
        echo ": $output_file1" | tee -a "$output_log"
    fi

    #  output_file2
    if [[ -f "$output_file2" ]]; then
        add_command "../../../exp_tool/kissat/kissat-4.0.1-linux-amd64 --unsat \"$output_file2\""
    else
        echo ": $output_file2" | tee -a "$output_log"
    fi

    #  output_file3
    if [[ -f "$output_file3" ]]; then
        add_command "../../../exp_tool/kissat/kissat-4.0.1-linux-amd64 --unsat \"$output_file3\""
    else
        echo ": $output_file3" | tee -a "$output_log"
    fi

    # AIGabc
    if [[ -f "$aig_file1" ]]; then
        add_command "./abc -c \"miter '$aig_file' '$aig_file1';&get;&fraig -y -v;&ps;time\""
        add_command "./abc -c \"&cec '$aig_file' '$aig_file1'\""
    else
        echo "AIG: $aig_file1" | tee -a "$output_log"
    fi

    if [[ -f "$aig_file2" ]]; then
        add_command "./abc -c \"miter '$aig_file' '$aig_file2';&get;&fraig -y -v;&ps;time\""
        add_command "./abc -c \"&cec '$aig_file' '$aig_file2'\""
    else
        echo "AIG: $aig_file2" | tee -a "$output_log"
    fi

    if [[ -f "$aig_file3" ]]; then
        add_command "./abc -c \"miter '$aig_file' '$aig_file3';&get;&fraig -y -v;&ps;time\""
        add_command "./abc -c \"&cec '$aig_file' '$aig_file3'\""
    else
        echo "AIG: $aig_file3" | tee -a "$output_log"
    fi

    if [[ ${#commands[@]} -gt 0 ]]; then
        {
            printf "%s\n" "${commands[@]}" | \
            parallel --halt now,done=1 --jobs "$num_jobs" > >(tee -a "$output_log") 2>&1
        }

        # Aggregate memory usage
        total_memory=0
        for temp_file in "${temp_files[@]}"; do
            max_mem=$(grep "Maximum resident set size" "$temp_file" | awk '{print $6}')
            total_memory=$((total_memory + max_mem))
            rm "$temp_file"
        done

        echo "Total memory usage: $total_memory KB" | tee -a "$output_log"
        echo ": $output_sub_dir" | tee -a "$output_log"
    else
        echo "" | tee -a "$output_log"
        echo ": $output_sub_dir"
    fi
}

# Read and process each row of the CSV
# CSV
tail -n +2 "$input_file" | while IFS=',' read -r col1 col2 col3 col4; do
    # Trim whitespace (optional but recommended)
    col1=$(echo "$col1" | xargs)
    col2=$(echo "$col2" | xargs)
    col3=$(echo "$col3" | xargs)
    col4=$(echo "$col4" | xargs)

    # Process the row in a subshell to handle errors without exiting the main script
    process_row "$col1" "$col2" "$col3" "$col4" || {
        echo "Error processing row: $col1, $col2, $col3, $col4" >> "$output_dir1/error.log"
    }
done

echo ": $output_dir2"