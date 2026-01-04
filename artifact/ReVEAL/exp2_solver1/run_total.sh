#!/bin/bash

scripts=(
    "cnf.sh"
    "abccec.sh"
    "abccec1.sh"
    "kissat.sh"
    "abcfraig.sh"
    "abcfraig1.sh"
)

for script in "${scripts[@]}"; do
    echo ": $script"
    
    script_dir=$(cd "$(dirname "$script")" && pwd)
    cd "$script_dir"
    
    if ./"$(basename "$script")"; then
        echo "$script "
    else
        echo "$script "
        exit 1
    fi
done

echo ""