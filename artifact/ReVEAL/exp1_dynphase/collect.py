import os
import re
import pandas as pd
import numpy as np  #  NumPy

base_directory = './'

for subdir in os.listdir(base_directory):
    subdir_path = os.path.join(base_directory, subdir)

    if os.path.isdir(subdir_path):
        output_csv = f'{subdir}.csv'  #  CSV 
        results = []  # 

        for filename in os.listdir(subdir_path):
            if filename.endswith('.aig'):
                aig_path = os.path.join(subdir_path, filename)
                
                #  .aig 
                with open(aig_path, 'r') as file:
                    content = file.readlines()
                
                circuit_status = None
                time_overall = None
                memory_usage = None
                
                #  CIRCUIT IS  Time overall
                for line in reversed(content):
                    if 'CIRCUIT IS' in line:
                        circuit_status = line.split()[-1].strip('.\n')
                    if 'Time overall was:' in line:
                        time_overall = line.split()[-1].strip()

                #  .txt  _output
                txt_filename = filename.replace('.aig', '').replace('_output', '') + '.txt'  
                txt_path = os.path.join(subdir_path, txt_filename)
                
                if os.path.exists(txt_path):
                    with open(txt_path, 'r') as txt_file:
                        txt_content = txt_file.readlines()  # 
                        
                        #  .txt 
                        if len(txt_content) < 3:  # 
                            memory_found = False
                            for line in txt_content:
                                line = line.strip()
                                if '' in line:
                                    memory_usage = 'MO'  # “”
                                    memory_found = True  # 
                                    break  # 
                            
                            if not memory_found:
                                time_overall = 'TO'  # 
                                print(f"Debug: Found 'TO' for {filename[:-4]}")  # 
                        else:
                            memory_usage_match = re.search(r'(\d+) KB', ''.join(txt_content))
                            if memory_usage_match:
                                memory_usage = memory_usage_match.group(1)

                print(f"Debug: File: {filename[:-4]}, Circuit Status: {circuit_status}, Time Overall: {time_overall}, Memory Usage: {memory_usage}")

                if memory_usage == 'MO':
                    time_overall = None  #  'MO' Time Overall  None
                elif time_overall == 'TO':
                    memory_usage = None  #  Time Overall  'TO' Memory Usage  None

                results.append([
                    filename[:-4],  #  .aig
                    circuit_status if circuit_status else 'UNKNOWN',  #  UNKNOWN
                    time_overall,  #  time_overall 'TO'  None
                    memory_usage if memory_usage else np.nan  #  Memory Usage np.nan 
                ])

        #  DataFrame
        df = pd.DataFrame(results, columns=['File Name', 'Circuit Status', 'Time Overall', 'Memory Usage'])

        # df['Time Overall'].fillna('TO', inplace=True)

        #  Time Overall 
        df['Time Overall'] = df['Time Overall'].astype(str)

        #  DataFrame
        print(df)

        #  CSV exp1 
        df = pd.DataFrame(results, columns=['File Name', 'Circuit Status', 'Time Overall', 'Memory Usage'])



        #  'TO' 
        df['Sort Time'] = df['Time Overall'].replace('TO', np.inf)

        #  float 
        df['Sort Time'] = pd.to_numeric(df['Sort Time'], errors='coerce')

        #  Sort Time 
        df.sort_values(by='Sort Time', inplace=True)

        #  Sort Time 
        df.drop(columns=['Sort Time'], inplace=True)

        #  Time Overall 
        df['Time Overall'] = df['Time Overall'].astype(str)

        #  DataFrame
        print(df)

        #  CSV exp1 
        output_file_path = os.path.join(base_directory, output_csv)
        try:
            df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
            print(f'Results saved to {output_file_path}')
        except Exception as e:
            print(f"Error saving file: {e}")