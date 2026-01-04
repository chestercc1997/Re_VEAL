import os
import re
import pandas as pd

# Define input and output directories
# Define input and output directories
input_dir = './resyn3_128_resultp'
output_csv = './timing_report_resyn3_128_newp.csv'

# Initialize results list
results = []

# Iterate over each subdirectory
for subdir in os.listdir(input_dir):
    subdir_path = os.path.join(input_dir, subdir)
    
    # Ensure it's a directory
    if os.path.isdir(subdir_path):
        output_file_path = os.path.join(subdir_path, 'output.txt')
        
        # Ensure the output.txt file exists
        if os.path.isfile(output_file_path):
            with open(output_file_path, 'r', encoding='utf-8') as file:
                content = file.readlines()
                
                # Initialize time, memory, and method variables
                time_seconds = None
                memory_kb = None
                method = None
                
                # Look for 'Networks are equivalent.  Time = ... sec' line
                for line in content:
                    if 'Networks are equivalent.  Time =' in line:
                        match = re.search(r'Time\s*=\s*([\d.]+)\s*sec', line)
                        if match:
                            time_seconds = match.group(1)
                            method = 'sat-sweeping'
                            break
                
                # If not found, look for 'c process-time:' line
                if time_seconds is None:
                    for line in content:
                        if 'c process-time:' in line:
                            match = re.search(r'c\s+process-time:.*?([\d.]+)\s+seconds', line)
                            if match:
                                time_seconds = match.group(1)
                                method = 'kissat'
                                break
                
                # If still not found, check 'UNSATISFIABLE' line
                if time_seconds is None:
                    for line in content:
                        if 'UNSATISFIABLE' in line:
                            match = re.search(r'Time\s*=\s*([\d.]+)\s*sec', line)
                            if match:
                                time_seconds = match.group(1)
                                method = 'glucose2'
                                break
                
                # NEW: If still not found, check 'total: ... seconds' line
                if time_seconds is None:
                    for line in content:
                        if 'total:' in line and 'seconds' in line:
                            match = re.search(r'total:\s*([\d.]+)\s*seconds', line)
                            if match:
                                time_seconds = match.group(1)
                                method = 'fraig'
                                break

                # Look for 'Total memory usage' line
                for line in content:
                    if 'Total memory usage:' in line:
                        match = re.search(r'Total memory usage:\s*([\d]+)\s*KB', line)
                        if match:
                            memory_kb = match.group(1)
                            break

                # Record the results
                if time_seconds is not None:
                    results.append([subdir, time_seconds, memory_kb, method])
                else:
                    results.append([subdir, 'N/A', memory_kb, 'N/A'])

# Create DataFrame
df = pd.DataFrame(results, columns=['File Name', 'Time (seconds)', 'Memory (KB)', 'Method'])

# Save as CSV
df.to_csv(output_csv, index=False, encoding='utf-8-sig')

# Print results saved information
print(f'Results saved to {output_csv}')