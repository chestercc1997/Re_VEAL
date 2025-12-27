import os
import re
import pandas as pd

# Define the base directory
base_directory = './'

# Traverse all subdirectories
for subdir in os.listdir(base_directory):
    subdir_path = os.path.join(base_directory, subdir)
    
    # Ensure it is a directory
    if os.path.isdir(subdir_path):
        output_csv = os.path.join(base_directory, f'{subdir}.csv')  # Output CSV file path
        results = []

        # Traverse all .txt files first within the subdirectory
        for filename in os.listdir(subdir_path):
            if filename.endswith('.txt'):
                txt_path = os.path.join(subdir_path, filename)
                
                # Initialize variables
                memory_usage = None
                circuit_status = 'UNKNOWN'  # Default status
                time_overall = None

                # Read .txt file content
                with open(txt_path, 'r', encoding='utf-8') as txt_file:
                    txt_content = txt_file.read()

                # Case 1: Successful run with memory usage and time
                # Example:
                # dynphaseorderopt 
                # 32616 KB
                # 2 
                peak_memory_match = re.search(r'(\d+)\s+KB', txt_content)
                runtime_match = re.search(r'(\d+)\s+', txt_content)

                if peak_memory_match:
                    memory_usage = peak_memory_match.group(1)
                    if runtime_match:
                        time_overall = runtime_match.group(1)
                else:
                    # Case 2: Memory usage exceeded 8GB and process terminated
                    # Example:
                    #  8GB8511564 KB: 123740
                    if re.search(r'', txt_content):
                        memory_usage = 'MO'
                    else:
                        # Case 3: Timeout without specific memory usage
                        time_overall = 'TO'

                # Determine the corresponding .aig filename by appending '_output.aig'
                base_txt_name = filename[:-4]  # Remove '.txt'
                aig_filename = f"{base_txt_name}_output.aig"
                aig_path = os.path.join(subdir_path, aig_filename)

                # Initialize variables for .aig file
                overall_runtime = None
                verification_result = 'UNKNOWN'

                # Process the corresponding .aig file if it exists
                if os.path.exists(aig_path):
                    with open(aig_path, 'r', encoding='utf-8') as aig_file:
                        aig_content = aig_file.read()

                    # Extract Overall run-time
                    runtime_pattern = re.search(r'total process time:\s+(\d+\.\d+)\s+seconds', aig_content)
                    if runtime_pattern:
                        overall_runtime = runtime_pattern.group(1)

                    # Extract Verification result (buggy or correct)
                    if 'buggy' in aig_content.lower():
                        verification_result = 'buggy'
                    elif 'correct' in aig_content.lower():
                        verification_result = 'correct'

                    # If time_overall is not 'TO', update it with overall_runtime
                    if time_overall != 'TO':
                        time_overall = overall_runtime if overall_runtime else time_overall

                    # Update circuit_status only if verification_result is determined
                    if verification_result in ['buggy', 'correct']:
                        circuit_status = verification_result
                else:
                    # If the .aig file is missing
                    circuit_status = 'AIG FILE MISSING'

                # Record the result
                results.append([
                    base_txt_name,          # File Name (without .txt)
                    circuit_status,         # Circuit Status
                    time_overall,           # Time Overall
                    memory_usage            # Memory Usage
                ])

        # If no .txt files found, add a row indicating this
        if not any(fname.endswith('.txt') for fname in os.listdir(subdir_path)):
            results.append(['No .txt files found', 'UNKNOWN', None, None])

        # Create DataFrame and sort
        df = pd.DataFrame(results, columns=['File Name', 'Circuit Status', 'Time Overall', 'Memory Usage'])

        # Convert 'Time Overall' to numeric for sorting, coerce errors to NaN
        df['Time Overall Numeric'] = pd.to_numeric(df['Time Overall'], errors='coerce')
        
        # Sort by 'Time Overall', placing NaN at the end
        df.sort_values(by='Time Overall Numeric', inplace=True, ascending=True)

        # Drop the helper numeric column
        df.drop(columns=['Time Overall Numeric'], inplace=True)

        # Save to CSV
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')

        print(f'Results saved to {output_csv}')