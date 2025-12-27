#!/bin/bash

resyn3_dir="../64_resyn3"
default_dir="../64_default"
output_dir="./resyn3/resyn3_64_result_fraig" # 
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

    if (( num_parts < 3 )); then
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

    output_file1="$output_dir/$aig_filename.txt"

    #  timeout "TO" ./abc 
    cmd="bash -c 'timeout 3600 ./abc -c \"miter $aig_file_path $matched_file;&get;&fraig -y -v;&ps;time\" > \"$output_file1\"; exit_code=\$?; if [ \$exit_code -eq 124 ]; then echo \"TO\" > \"$output_file1\"; fi'"

    echo "$cmd" >> "$command_log"
done

if [[ ! -s "$command_log" ]]; then
    echo ""
    exit 0
fi

#  GNU Parallel 
#  -j  64
parallel -j 64 < "$command_log"

rm "$command_log"

echo ""


