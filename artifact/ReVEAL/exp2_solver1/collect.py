import os
import re
import csv

#  CSV 
input_dir = './resyn3'
output_dir = input_dir  # CSV  dc2 

def extract_time(file_path):
    time_seconds = None
    with open(file_path, 'r') as file:
        content = file.readlines()
        
        # : 'Networks are equivalent.  Time ='
        for line in content:
            if 'Networks are equivalent.  Time =' in line:
                match = re.search(r'Time\s*=\s*([\d.]+)\s*sec', line)
                if match:
                    time_seconds = match.group(1)
                    return time_seconds
        
        # : 'TOTAL         ='
        for line in content:
            if 'TOTAL         =' in line:
                match = re.search(r'TOTAL\s*=\s*([\d.]+)\s*sec', line)
                if match:
                    time_seconds = match.group(1)
                    return time_seconds
        
        # : 'c process-time:'
        for line in content:
            if 'c process-time:' in line:
                match = re.search(r'c\s+process-time:.*?([\d.]+)\s+seconds', line)
                if match:
                    time_seconds = match.group(1)
                    return time_seconds
    
    #  "TO"Timeout
    if time_seconds is None:
        return "TO"

for subdir in os.listdir(input_dir):
    subdir_path = os.path.join(input_dir, subdir)
    
    if os.path.isdir(subdir_path):
        #  CSV  dc2 
        output_csv = os.path.join(output_dir, f"{subdir}.csv")
        results = []  # 
        
        for file in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, file)
            
            if os.path.isfile(file_path):
                time = extract_time(file_path)
                #  "TO"
                numeric_time = float(time) if time != "TO" else float('inf')
                results.append({'filename': file, 'time': time, 'numeric_time': numeric_time})
        
        #  numeric_time 
        results.sort(key=lambda x: x['numeric_time'])

        #  dc2  CSV 
        with open(output_csv, 'w', newline='') as csvfile:
            csv_header = ['filename', 'time']
            writer = csv.DictWriter(csvfile, fieldnames=csv_header)
            writer.writeheader()  # 
            for result in results:
                writer.writerow({'filename': result['filename'], 'time': result['time']})
        
        print(f"Results for {subdir} written to {output_csv}")