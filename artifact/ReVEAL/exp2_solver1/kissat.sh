#!/bin/bash

cnf_dir="./cnf_resyn3_64"
result_dir="./resyn3/resyn3_64_result_kissat"
kissat_path="../../../exp_tool/kissat/kissat-4.0.1-linux-amd64"
command_log="./kissat_commands.sh"

mkdir -p "$result_dir"
# rm -f "$result_dir"/*.result

#  kissat 
if [[ ! -x "$kissat_path" ]]; then
    echo ": kissat : $kissat_path"
    exit 1
fi

> "$command_log"

#  cnf_dir  .cnf 
shopt -s nullglob
cnf_files=("$cnf_dir"/*.cnf)
shopt -u nullglob

if (( ${#cnf_files[@]} == 0 )); then
    echo " $cnf_dir  .cnf "
    exit 0
fi

for cnf_file_path in "${cnf_files[@]}"; do
    cnf_filename=$(basename "$cnf_file_path" .cnf)
    echo ": $cnf_filename.cnf"

    result_file="$result_dir/$cnf_filename.result"

    #  bash -c 
    #  timeout  "TO"
    cmd="bash -c 'timeout 3600 \"$kissat_path\" --unsat \"$cnf_file_path\" > \"$result_file\"; exit_code=\$?; if [ \$exit_code -eq 124 ]; then echo \"TO\" > \"$result_file\"; fi'"

    echo "$cmd" >> "$command_log"
done

if [[ ! -s "$command_log" ]]; then
    echo ""
    exit 0
fi

#  GNU Parallel 
#  4  -j 
echo " kissat ..."
parallel -j 64 < "$command_log"

rm "$command_log"

echo " $result_dir "