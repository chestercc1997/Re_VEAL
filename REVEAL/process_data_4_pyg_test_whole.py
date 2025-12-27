import os
import shutil
import gzip
import pandas as pd
import argparse

def main(op, lmsb):
    source_dir = f"./data_4_ml_train_{op}_{lmsb}_whole"
    target_dir_train = f"./dataset_ml_train_{op}_{lmsb}_whole"
    source_dir_test = f"./data_4_ml_test_{op}_{lmsb}_whole"
    target_dir_test = f"./dataset_ml_test_{op}_{lmsb}_whole"

    for target_dir in [target_dir_train, target_dir_test]:
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)

    for target_dir in [target_dir_train, target_dir_test]:
        os.makedirs(target_dir)

    for folder in sorted(os.listdir(source_dir)):
        source_folder = os.path.join(source_dir, folder)
        if os.path.isdir(source_folder):
            for subfolder in ["class_map", "edgelist", "feature"]:
                source_subfolder = os.path.join(source_folder, subfolder)
                if os.path.isdir(source_subfolder):
                    for filename in sorted(os.listdir(source_subfolder)):
                        source_file = os.path.join(source_subfolder, filename)
                        target_folder = os.path.join(target_dir_train, f"{folder}", "raw")
                        os.makedirs(target_folder, exist_ok=True)
                        target_file = os.path.join(target_folder, filename)
                        shutil.copy(source_file, target_file)

    for folder in sorted(os.listdir(source_dir_test)):
        source_folder = os.path.join(source_dir_test, folder)
        if os.path.isdir(source_folder):
            for subfolder in ["class_map", "edgelist", "feature"]:
                source_subfolder = os.path.join(source_folder, subfolder)
                if os.path.isdir(source_subfolder):
                    for filename in sorted(os.listdir(source_subfolder)):
                        source_file = os.path.join(source_subfolder, filename)
                        target_folder = os.path.join(target_dir_test, f"{folder}", "raw")
                        os.makedirs(target_folder, exist_ok=True)
                        target_file = os.path.join(target_folder, filename)
                        shutil.copy(source_file, target_file)

    dataset_dirs = [target_dir_train, target_dir_test]
    process_dataset(dataset_dirs)

def process_dataset(dataset_dirs):
    for dataset_dir in dataset_dirs:
        for folder in os.listdir(dataset_dir):
            folder_path = os.path.join(dataset_dir, folder)
            if os.path.isdir(folder_path):
                process_folder(folder_path, folder)

def process_folder(folder_path, folder_name):
    raw_folder = os.path.join(folder_path, "raw")
    processed_folder = os.path.join(folder_path, "processed")
    
    if os.path.isdir(raw_folder):
        os.makedirs(processed_folder, exist_ok=True)
        processed_files = set()
        
        for filename in os.listdir(raw_folder):
            source_file = os.path.join(raw_folder, filename)
            if filename.endswith(".json"):
                new_filename = "node-feat.csv"
                if filename != new_filename and new_filename not in processed_files:
                    df = pd.read_csv(source_file, header=None)
                    target_file = os.path.join(raw_folder, new_filename)
                    df.to_csv(target_file, index=False)
                    os.rename(source_file, target_file)
                    processed_files.add(new_filename)
                num_nodes = count_lines(target_file)
                num_node_file = os.path.join(raw_folder, "num-node-list.csv")
                with open(num_node_file, "w") as f:
                    f.write(str(num_nodes))

            elif filename.endswith(".txt"):
                if filename == "class_map.txt":
                    new_filename = "graph-label.csv"
                    if filename != new_filename and new_filename not in processed_files:
                        target_file = os.path.join(raw_folder, new_filename)
                        shutil.copy(source_file, target_file)
                        os.remove(source_file)
                        processed_files.add(new_filename)
                else:
                    new_filename = "graph-feature.csv"
                    if filename != new_filename and new_filename not in processed_files:
                       target_file = os.path.join(raw_folder, new_filename)
                       shutil.copy(source_file, target_file)
                       os.remove(source_file)
                       processed_files.add(new_filename)


            elif filename.endswith(".el"):
                new_filename = "edge.csv"
                if filename != new_filename and new_filename not in processed_files:
                    target_file = os.path.join(raw_folder, new_filename)
                    edge_data = []
                    with open(source_file, "r") as f:
                        for line in f:
                            source, target = line.strip().split()
                            edge_data.append(f"{source},{target}")
                    with open(target_file, "w", newline="") as csvfile:
                        for line in edge_data:
                            csvfile.write(line + "\n")
                    os.remove(source_file)
                    processed_files.add(new_filename)
                num_edges = count_lines(target_file)
                num_edge_file = os.path.join(raw_folder, "num-edge-list.csv")
                with open(num_edge_file, "w") as f:
                    f.write(str(num_edges))

        for filename in os.listdir(raw_folder):
            if filename.endswith(".csv"):
                csv_file = os.path.join(raw_folder, filename)
                gz_file = os.path.join(raw_folder, f"{filename}.gz")
                with open(csv_file, "rb") as f_in, gzip.open(gz_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

def count_lines(file_path):
    with open(file_path, "r") as f:
        return sum(1 for _ in f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process datasets.')
    parser.add_argument('--op', type=str, default='default', help='Operation type (default: "default")')
    parser.add_argument('--lmsb', type=str, choices=['MSB', 'LSB','LSB1'], default='LSB', help='Specify the type: "MSB" or "LSB" (default: "LSB")')
    
    args = parser.parse_args()
    
    main(args.op, args.lmsb)