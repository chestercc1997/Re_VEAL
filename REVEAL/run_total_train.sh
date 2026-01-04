#!/bin/bash
start_time=$(date +%s)

# ==============================================================================
# parameters
# ==============================================================================
cut_level=5  # Set the cut_level value as needed
outputsize=8
lmsb1="LSB"  # LSB model
lmsb2="MSB"  # MSB model
lmsb3="LSB1"  # LSB1 model

# op0="resyn3_U"
# op="resyn3_U"
op0="dc2_U"
op="dc2_U"

task=1
mode="train"  # Set to "train" mode to train the model
hoga=0
batchsize=4096
device_number=1

# ==============================================================================
# Define dataset paths
# ==============================================================================
root_path_train_lsb1="dataset_ml_train_${op}_${lmsb3}"
root_path_test_lsb1="dataset_ml_test_${op}_${lmsb3}"
root_path_train_lsb="dataset_ml_train_${op0}_${lmsb1}"
root_path_test_lsb="dataset_ml_test_${op0}_${lmsb1}"
root_path_train_msb="dataset_ml_train_${op0}_${lmsb2}"
root_path_test_msb="dataset_ml_test_${op0}_${lmsb2}"

# ==============================================================================
# Define data sizes for training and testing
# ==============================================================================
# for size in {32..63}; do
#     data_sizes_train+=("$size")
# done
for size in {32..63}; do
    if (( size % 2 == 0 )); then  # Check if the number is even
        data_sizes_train+=("$size")  # If even, add to the array
    fi
done
data_sizes_test=("64" "128" "256")

# ==============================================================================
# Step 1: Graph partition for feature extraction
# ==============================================================================
echo "========================================"
echo "Step 1: Graph partition for feature extraction"
echo "========================================"
cd ..
bash exp_tool/LSB_MSB_cone_extact_all.sh "$lmsb1" "$op" "$cut_level" "$outputsize" --train "${data_sizes_train[@]}" --test "${data_sizes_test[@]}"
bash exp_tool/LSB_MSB_cone_extact_all.sh "$lmsb2" "$op" "$cut_level" "$outputsize" --train "${data_sizes_train[@]}" --test "${data_sizes_test[@]}"

end_time1=$(date +%s)

# ==============================================================================
# Step 2: Word-level and graph feature extraction & labeling
# ==============================================================================
echo "========================================"
echo "Step 2: Word-level and graph feature extraction & labeling"
echo "========================================"
bash REVEAL/dataset1_ml_test_total_amg.sh "$lmsb1" "$op" --train "${data_sizes_train[@]}" --test "${data_sizes_test[@]}"
bash REVEAL/dataset1_ml_test_total_amg.sh "$lmsb2" "$op" --train "${data_sizes_train[@]}" --test "${data_sizes_test[@]}"

# ==============================================================================
# Step 3: Package the data for training & testing
# ==============================================================================
echo "========================================"
echo "Step 3: Package the data for training & testing"
echo "========================================"
cd REVEAL
python process_data_4_pyg_test.py --op "$op" --lmsb "$lmsb1"
python process_data_4_pyg_test.py --op "$op" --lmsb "$lmsb2"
# python process_data_4_pyg_test_hoga.py --op "$op" --lmsb "$lmsb1"
# python process_data_4_pyg_test_hoga.py --op "$op" --lmsb "$lmsb2"

# ==============================================================================
# Step 4: Training & testing
# ==============================================================================
echo "========================================"
echo "Step 4: Training & testing"
echo "========================================"
# python main_optuna.py --pretrain 0 --task "$task" --mode "$mode" --op "$op" --hoga "$hoga" --device "$device_number"  --batch_size "$batchsize"\
#      --root_path_train_lsb1 "$root_path_train_lsb1" --root_path_test_lsb1 "$root_path_test_lsb1" \
#      --root_path_train_lsb "$root_path_train_lsb" --root_path_test_lsb "$root_path_test_lsb" \
#      --root_path_train_msb "$root_path_train_msb" --root_path_test_msb "$root_path_test_msb"\
#      --root_path_train_msb_hoga "$root_path_train_msb_hoga" --root_path_test_msb_hoga "$root_path_test_msb_hoga"
python main_arc_predict.py --pretrain 0 --task "$task" --mode "$mode" --op "$op" --device "$device_number"  --batch_size "$batchsize"\
     --root_path_train_lsb1 "$root_path_train_lsb1" --root_path_test_lsb1 "$root_path_test_lsb1" \
     --root_path_train_lsb "$root_path_train_lsb" --root_path_test_lsb "$root_path_test_lsb" \
     --root_path_train_msb "$root_path_train_msb" --root_path_test_msb "$root_path_test_msb"

# ==============================================================================
# Summary: Calculate and output the runtime
# ==============================================================================
end_time=$(date +%s)

runtime=$((end_time - start_time))
runtime0=$((end_time1 - start_time))
echo "========================================"
echo "Execution Summary"
echo "========================================"
echo "Feature extraction runtime (Step 1): $runtime0 seconds"
echo "Total runtime: $runtime seconds"

# python process_data_4_pyg_test.py --op "$op" --lmsb "$lmsb3"
# bash REVEAL/dataset_ml_test_total.sh "$lmsb3" "$op" "${data_sizes_train[@]}" "${data_sizes_test[@]}"