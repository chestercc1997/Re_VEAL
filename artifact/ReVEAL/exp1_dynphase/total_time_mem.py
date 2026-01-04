import os
import pandas as pd
import glob

# Define the path pattern for the CSV files
file_pattern = '../exp1/*_output.csv'

# Initialize sums and counts
total_time = 0
total_memory = 0
time_count = 0
memory_count = 0

# Constants for "TO" and "MO"
TO_TIME_SECONDS = 3600  # Adjust as needed
MO_MEMORY_KB = 8 * 1024 * 1024  # 8 GB in KB

# Iterate over each CSV file
for csv_file in glob.glob(file_pattern):
    df = pd.read_csv(csv_file)

    # Ensure the relevant columns exist
    if 'Time Overall' in df.columns and 'Memory Usage' in df.columns:
        # Adjust counts for TO and MO
        time_adjustments = df['Memory Usage'].apply(lambda x: 1 if x == 'MO' else 0).sum()
        memory_adjustments = df['Time Overall'].apply(lambda x: 1 if x == 'TO' else 0).sum()

        time_count += len(df) - time_adjustments
        memory_count += len(df) - memory_adjustments

        # Replace "TO" and "MO" with their numeric values
        df['Time Overall'] = df['Time Overall'].replace('TO', TO_TIME_SECONDS)
        df['Memory Usage'] = df['Memory Usage'].replace('MO', MO_MEMORY_KB)

        # Convert to numeric, ignoring any non-convertible entries
        df['Time Overall'] = pd.to_numeric(df['Time Overall'], errors='coerce')
        df['Memory Usage'] = pd.to_numeric(df['Memory Usage'], errors='coerce')

        # Sum the Time Overall and Memory Usage for the current file
        file_time_sum = df['Time Overall'].sum(skipna=True)
        file_memory_sum = df['Memory Usage'].sum(skipna=True)

        # Add to the total sums
        total_time += file_time_sum
        total_memory += file_memory_sum

# Calculate averages
average_time = total_time / time_count if time_count > 0 else 0
average_memory = total_memory / memory_count if memory_count > 0 else 0

# Print the total sums, averages, and counts
print(f'Total Time (seconds): {total_time}')
print(f'Total Memory (KB): {total_memory}')
print(f'Average Time (seconds): {average_time}')
print(f'Average Memory (KB): {average_memory}')
print(f'Time Count: {time_count}')
print(f'Memory Count: {memory_count}')