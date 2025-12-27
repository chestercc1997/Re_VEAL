#!/bin/bash
#===============================================================================
# Activation script for PyG environment
# Usage: source env/activate_env.sh
#===============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Environment name
ENV_NAME="pyg_env"

#===============================================================================
# Initialize Conda
#===============================================================================
init_conda() {
    # Try to find and source conda
    if [[ -n "$CONDA_EXE" ]]; then
        # Conda is already initialized
        CONDA_BASE=$(dirname $(dirname "$CONDA_EXE"))
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
        source "/opt/conda/etc/profile.d/conda.sh"
    elif [[ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "/usr/local/anaconda3/etc/profile.d/conda.sh"
    else
        echo "[ERROR] Could not find conda installation."
        echo "Please ensure conda is installed and try again."
        return 1
    fi
}

#===============================================================================
# Set Environment Variables
#===============================================================================
set_env_vars() {
    # Python settings
    export PYTHONUNBUFFERED=1
    export PYTHONDONTWRITEBYTECODE=1
    
    # Cache directories
    export TRANSFORMERS_CACHE="${HOME}/.cache/huggingface/hub"
    export HF_HOME="${HOME}/.cache/huggingface"
    export MPLCONFIGDIR="${HOME}/.config/matplotlib"
    export TORCH_HOME="${HOME}/.cache/torch"
    
    # Create cache directories if they don't exist
    mkdir -p "$TRANSFORMERS_CACHE" "$MPLCONFIGDIR" "$TORCH_HOME" 2>/dev/null
    
    # CUDA settings (optional, uncomment if needed)
    # export CUDA_VISIBLE_DEVICES=0
    # export CUDA_LAUNCH_BLOCKING=1  # For debugging
}

#===============================================================================
# Print Environment Info
#===============================================================================
print_env_info() {
    echo ""
    echo "==============================================================================="
    echo " PyG Environment Activated"
    echo "==============================================================================="
    echo ""
    echo "  Environment:  $ENV_NAME"
    echo "  Python:       $(which python)"
    echo "  Python Ver:   $(python --version 2>&1 | awk '{print $2}')"
    echo ""
    
    # Check PyTorch and CUDA
    python -c "
import torch
print(f'  PyTorch:      {torch.__version__}')
print(f'  CUDA Built:   {torch.version.cuda}')
print(f'  CUDA Avail:   {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU Count:    {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}:        {props.name} ({props.total_memory / 1024**3:.1f} GB)')
" 2>/dev/null || echo "  [Warning] Could not import torch"
    
    # Check PyG
    python -c "
import torch_geometric
print(f'  PyG:          {torch_geometric.__version__}')
" 2>/dev/null || echo "  [Warning] Could not import torch_geometric"
    
    echo ""
    echo "==============================================================================="
    echo ""
}

#===============================================================================
# Main
#===============================================================================
main() {
    # Initialize conda
    init_conda || return 1
    
    # Check if environment exists
    if ! conda env list | grep -q "^${ENV_NAME} "; then
        echo "[ERROR] Environment '$ENV_NAME' not found."
        echo "Please run ./env/setup_env.sh first to create the environment."
        return 1
    fi
    
    # Activate environment
    conda activate "$ENV_NAME"
    
    if [[ $? -ne 0 ]]; then
        echo "[ERROR] Failed to activate environment '$ENV_NAME'"
        return 1
    fi
    
    # Set environment variables
    set_env_vars
    
    # Print info
    print_env_info
}

# Run main function
main