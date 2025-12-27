#!/bin/bash

input_file="../../../REVEAL/dump/pred_stage3_resyn3_U.txt"
# # 
output_base="./pred_stage3_resyn3_U_"
declare -A reverse_mapping
reverse_mapping=( ["0"]="RC" ["1"]="SE" ["2"]="CL" ["3"]="CK" ["4"]="HCA" ["5"]="LF" ["6"]="KS" ["7"]="BK" ["8"]="JCA" )

temp_file="./temp_output.csv"
echo "Filename,Column2,Column3,Column4" > "$temp_file"

while IFS=',' read -r line; do
    filename=$(echo "$line" | cut -d',' -f1)
    col2=$(echo "$line" | cut -d',' -f2)
    col3=$(echo "$line" | cut -d',' -f3)
    col4=$(echo "$line" | cut -d',' -f4)

    mapped_col2=${reverse_mapping[$col2]}
    mapped_col3=${reverse_mapping[$col3]}
    mapped_col4=${reverse_mapping[$col4]}

    #  IFS  read 
    IFS='_' read -r -a parts <<< "$filename"

    base_filename="${parts[@]:0:${#parts[@]}-3}"  # 
    base_filename=$(echo "$base_filename" | tr ' ' '_')  # 

    #  base_filename  "SP" 
    if [[ "$base_filename" == *"SP"* ]]; then
        new_filename_col2="${base_filename}_${mapped_col2}_GenMul_default"
        new_filename_col3="${base_filename}_${mapped_col3}_GenMul_default"
        new_filename_col4="${base_filename}_${mapped_col4}_GenMul_default"
    else
        new_filename_col2="${base_filename}_${mapped_col2}_Multgen_default"
        new_filename_col3="${base_filename}_${mapped_col3}_Multgen_default"
        new_filename_col4="${base_filename}_${mapped_col4}_Multgen_default"
    fi

    echo "$filename,$new_filename_col2,$new_filename_col3,$new_filename_col4" >> "$temp_file"
done < "$input_file"

sort -t',' -k1,1 -V "$temp_file" > "$temp_file.sorted"

declare -A output_files
while IFS=',' read -r filename new_col2 new_col3 new_col4; do
    first_number=$(echo "$filename" | grep -oP '\d+' | head -n 1)

    output_file="${output_base}${first_number}.csv"

    if [[ ! -f "$output_file" ]]; then
        echo "Filename,Column2,Column3,Column4" > "$output_file"
    fi

    #  CSV 
    echo "$filename,$new_col2,$new_col3,$new_col4" >> "$output_file"
done < "$temp_file.sorted"

rm "$temp_file" "$temp_file.sorted"

echo ": ${output_base}64.csv, ${output_base}128.csv, ${output_base}256.csv"