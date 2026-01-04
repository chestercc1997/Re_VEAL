set -e

# Define maximum parallel jobs (64 cores)
MAX_JOBS=64

# Define memory and time limits
MEMORY_LIMIT=$((8 * 1024 * 1024))  # 8GB in KB
TIME_LIMIT=3600                    # 3600 seconds = 1 hour

# Check if GNU Parallel is installed
if ! command -v parallel &>/dev/null; then
    echo "Error: GNU parallel is not installed. Please install it using 'sudo apt-get install parallel' and run the script again."
    exit 1
fi

# Function to process a single .aig file
process_aig() {
    local aig_file="$1"
    local output_dir="$2"

    # Extract base filename (without path and extension)
    local base_name
    base_name=$(basename "$aig_file" .aig)

    # Define output log file path
    local log_file="$output_dir/${base_name}.txt"

    # Initialize peak memory and elapsed time
    local peak_memory=0
    local elapsed_time=0
    output_aig_file="${output_dir}/${base_name}_output.aig"

    # Run dynphaseorderopt, output results to specified file
    ./revsca "$aig_file" "$output_aig_file" -u &
    local pid=$!

    # Record task start time
    local start_time=$(date +%s)

    # Monitor memory usage and runtime
    while kill -0 "$pid" 2>/dev/null; do
        # Get current RSS memory usage (KB)
        current_memory=$(ps -o rss= -p "$pid" | tr -d ' ')

        # Check if memory value was successfully obtained
        if [[ -z "$current_memory" || ! "$current_memory" =~ ^[0-9]+$ ]]; then
            echo "Error: Unable to get memory usage for process $pid, current value: '$current_memory'. Attempting to terminate process." >> "$log_file"
            kill -9 "$pid" 2>/dev/null
            return
        fi

        # Update peak memory
        if (( current_memory > peak_memory )); then
            peak_memory=$current_memory
        fi

        # Calculate elapsed time
        local current_time=$(date +%s)
        elapsed_time=$((current_time - start_time))

        # Check if memory exceeds limit
        if (( current_memory > MEMORY_LIMIT )); then
            echo "Memory usage exceeded 8GB (${current_memory} KB), terminating process: $pid." >> "$log_file"
            kill -9 "$pid" 2>/dev/null
            return
        fi

        # Check if time exceeds limit
        if (( elapsed_time >= TIME_LIMIT )); then
            echo "Runtime exceeded 3600 seconds (1 hour), terminating process: $pid (time out)." >> "$log_file"
            kill -9 "$pid" 2>/dev/null
            return
        fi

        # Check every second
        sleep 1
    done

    # Wait for dynphaseorderopt process to complete and get exit status
    wait "$pid"
    local exit_status=$?

    # Log information based on exit status
    if (( exit_status != 0 )); then
        echo "Error: dynphaseorderopt exited with non-zero status, exit code: $exit_status." >> "$log_file"
    else
        echo "Success: dynphaseorderopt completed successfully." >> "$log_file"
    fi

    # Log peak memory usage and total runtime
    echo "Peak memory usage: ${peak_memory} KB" >> "$log_file"
    echo "Total runtime: ${elapsed_time} seconds" >> "$log_file"
}

# Export function for parallel to use
export -f process_aig

# Export variables for parallel to use
export MEMORY_LIMIT
export TIME_LIMIT

# Define input and output directory arrays using relative paths
declare -a dirs=(
    "../../ReVEAL/256_dc2/:../../ReVEAL/exp1_dynphase/dc2_256_output"
    "../../ReVEAL/256_resyn3/:../../ReVEAL/exp1_dynphase/resyn3_256_output"
    "../../ReVEAL/128_dc2/:../../ReVEAL/exp1_dynphase/dc2_128_output"
    "../../ReVEAL/128_resyn3/:../../ReVEAL/exp1_dynphase/resyn3_128_output"
)

# Process each directory
for dir in "${dirs[@]}"; do
    IFS=':' read -r INPUT_DIR OUTPUT_DIR <<< "$dir"

    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"

    # Find all .aig files and process them in parallel using GNU Parallel
    find "$INPUT_DIR" -type f -name "*.aig" | parallel -j "$MAX_JOBS" process_aig {} "$OUTPUT_DIR"
done

echo "All files processed successfully!"