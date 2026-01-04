#!/bin/bash

resyn3_dir="../64_resyn3"
default_dir="../64_default"
output_dir="./cnf_resyn3_64"
command_log="./commands.sh"

mkdir -p "$output_dir"

> "$command_log"

#  resyn3_dir  .aig 
for aig_file_path in "$resyn3_dir"/*.aig; do
    #  .aig 
    if [[ ! -e "$aig_file_path" ]]; then
        echo " .aig  $resyn3_dir"
        exit 1
    fi

    aig_filename=$(basename "$aig_file_path" .aig)
    echo ": $aig_filename"  # 

    IFS='_' read -r -a parts <<< "$aig_filename"

    num_parts=${#parts[@]}

    if (( num_parts < 2 )); then
        echo ": $aig_filename"
        continue
    fi

    #  'RC'
    parts[$((num_parts - 3))]="RC"

    #  'default'
    parts[$((num_parts - 1))]="default"

    new_filename=$(IFS='_'; echo "${parts[*]}")

    echo ": $new_filename"

    matched_file="$default_dir/$new_filename.aig"

    if [[ ! -f "$matched_file" ]]; then
        echo ": $matched_file"
        continue
    fi

    #  CNF 
    output_file1="$output_dir/$aig_filename.cnf"

    cmd="./abc -c \"read $aig_file_path; miter $aig_file_path $matched_file; st; ps; write_cnf $output_file1\""

    echo "$cmd" >> "$command_log"
done

if [[ ! -s "$command_log" ]]; then
    echo ""
    exit 0
fi

#  GNU Parallel 
#  4  -j 
parallel -j 4 < "$command_log"

rm "$command_log"

echo ""