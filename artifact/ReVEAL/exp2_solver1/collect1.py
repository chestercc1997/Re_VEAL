import os
import csv
from collections import defaultdict

input_dir = './resyn3_new'

#  case  runtime 
case_times = defaultdict(list)  # key: case name, value: list of (method, time)
method_wins = defaultdict(int)  # key: method, value: count of wins

#  CSV 
for csv_file in os.listdir(input_dir):
    csv_path = os.path.join(input_dir, csv_file)

    #  .csv 
    if os.path.isfile(csv_path) and csv_file.endswith('.csv'):
        method_name = csv_file  # 
        with open(csv_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                raw_filename = row['filename']  # 
                time_str = row['time']

                #  "TO" 3600 
                if time_str.strip() == "TO":
                    time = 3600.0
                else:
                    time = float(time_str)

                #  `.result`  `.txt`
                if raw_filename.endswith('.result'):
                    case_name = raw_filename.replace('.result', '.txt')
                else:
                    case_name = raw_filename

                #  case_times 
                case_times[case_name].append((method_name, time))

#  case 
for case, times in case_times.items():
    #  runtime 
    times.sort(key=lambda x: x[1])  # x[1] 
    best_method = times[0][0]  # 
    method_wins[best_method] += 1

print(" case ")
for method, wins in method_wins.items():
    print(f"{method}: {wins} ")

#  case 
print("\n case ")
for case, times in case_times.items():
    best_method = min(times, key=lambda x: x[1])[0]
    print(f"{case}: {best_method}")