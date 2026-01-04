#!/bin/bash

# input_file="../our/pred_stage3_resyn3_U_64.csv"

input_file="./pred_stage3_resyn3_U_64.csv"
#  .aig 
resyn3_dir="../128_resyn3"
default_dir="../128_default"
output_dir1="./cnf_resyn3_128"

mkdir -p "$output_dir1"

missing_log="$output_dir1/missing_files.log"
: > "$missing_log"

commands=()
while IFS=',' read -r col1 col2 col3 col4; do
    #  .aig 
    aig_file="$resyn3_dir/$col1.aig"
    aig_file1="$default_dir/$col2.aig"
    aig_file2="$default_dir/$col3.aig"
    aig_file3="$default_dir/$col4.aig"

    #  col1 
    col1_dir="$output_dir1/$col1"
    mkdir -p "$col1_dir"

    output_file1="$col1_dir/cnf1.cnf"
    output_file2="$col1_dir/cnf2.cnf"
    output_file3="$col1_dir/cnf3.cnf"

    command_found=false

    if [[ -f "$aig_file" ]]; then
        command_found=true
        commands+=(
            "./abc -c \"read $aig_file; miter $aig_file $aig_file1; st; ps; write_cnf $output_file1\""
        )
    else
        echo ": $aig_file" >> "$missing_log"
    fi

    if [[ -f "$aig_file1" ]]; then
        command_found=true
        commands+=(
            "./abc -c \"read $aig_file; miter $aig_file $aig_file1; st; ps; write_cnf $output_file1\""
        )
    else
        echo ": $aig_file1" >> "$missing_log"
    fi

    if [[ -f "$aig_file2" ]]; then
        command_found=true
        commands+=(
            "./abc -c \"read $aig_file; miter $aig_file $aig_file2; st; ps; write_cnf $output_file2\""
        )
    else
        echo ": $aig_file2" >> "$missing_log"
    fi

    if [[ -f "$aig_file3" ]]; then
        command_found=true
        commands+=(
            "./abc -c \"read $aig_file; miter $aig_file $aig_file3; st; ps; write_cnf $output_file3\""
        )
    else
        echo ": $aig_file3" >> "$missing_log"
    fi

done < "$input_file"

#  parallel  --halt 
if [ ${#commands[@]} -gt 0 ]; then
    printf "%s\n" "${commands[@]}" | parallel -j3 --halt soon,fail=1
else
    echo ""
fi

echo ": $output_dir1"