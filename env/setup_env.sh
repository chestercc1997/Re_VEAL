#!/bin/bash
#===============================================================================
# PyG Environment Setup Script for Saturn Cloud
# Uses conda run to avoid conda activate issues
#===============================================================================

set -e

ENV_NAME="pyg_env"
PYTHON_VERSION="3.11"

# PyTorch and PyG versions (based on available wheels)
TORCH_VERSION="2.4.0"
TORCH_CUDA="cu121"
PYG_WHEEL_URL="https://data.pyg.org/whl/torch-2.4.0+cu121.html"

echo "==============================================================================="
echo " PyG Environment Setup for Saturn Cloud"
echo "==============================================================================="
echo ""
echo " PyTorch: ${TORCH_VERSION} + CUDA ${TORCH_CUDA}"
echo " Python:  ${PYTHON_VERSION}"
echo ""

#===============================================================================
# Install System Dependencies
#===============================================================================
echo "[Step 0/6] Installing system dependencies..."

# Check if we have sudo access
if command -v sudo &> /dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq bc time
else
    apt-get update -qq
    apt-get install -y -qq bc time
fi

echo "[OK] System dependencies installed"

#===============================================================================
# Initialize Conda
#===============================================================================
echo ""
echo "[Step 1/6] Initializing Conda..."

# Source conda
if [[ -f "/opt/saturncloud/etc/profile.d/conda.sh" ]]; then
    source /opt/saturncloud/etc/profile.d/conda.sh
elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    source /opt/conda/etc/profile.d/conda.sh
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "[ERROR] Cannot find conda.sh"
    exit 1
fi

echo "[OK] Conda initialized"

#===============================================================================
# Create or Recreate Environment
#===============================================================================
echo ""
echo "[Step 2/6] Creating Conda environment: $ENV_NAME..."

# Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[INFO] Removing existing environment..."
    conda env remove -n "$ENV_NAME" -y
fi

# Create new environment
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

echo "[OK] Environment created"

#===============================================================================
# Install PyTorch (using conda run)
#===============================================================================
echo ""
echo "[Step 3/6] Installing PyTorch ${TORCH_VERSION}+${TORCH_CUDA}..."

conda run -n "$ENV_NAME" pip install torch==${TORCH_VERSION} torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/${TORCH_CUDA}

# Verify PyTorch
echo "[INFO] Verifying PyTorch installation..."
conda run -n "$ENV_NAME" python -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"

echo "[OK] PyTorch installed"

#===============================================================================
# Install PyG
#===============================================================================
echo ""
echo "[Step 4/6] Installing PyTorch Geometric..."

# Install PyG main package
conda run -n "$ENV_NAME" pip install torch_geometric

# Install PyG extensions (using the provided wheel URL)
echo "[INFO] Installing PyG extensions from: $PYG_WHEEL_URL"
conda run -n "$ENV_NAME" pip install \
    pyg_lib \
    torch_scatter \
    torch_sparse \
    torch_cluster \
    torch_spline_conv \
    -f "$PYG_WHEEL_URL"

# Verify PyG
echo "[INFO] Verifying PyG installation..."
conda run -n "$ENV_NAME" python -c "
import torch_geometric
print(f'  PyG version: {torch_geometric.__version__}')
"

echo "[OK] PyG installed"

#===============================================================================
# Install Additional Dependencies
#===============================================================================
echo ""
echo "[Step 5/6] Installing additional dependencies..."

conda run -n "$ENV_NAME" pip install \
    numpy \
    scipy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    tqdm \
    tensorboard \
    ipykernel \
    optuna \
    optuna-integration

# Add to Jupyter
conda run -n "$ENV_NAME" python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

echo "[OK] Dependencies installed"

#===============================================================================
# Full Verification
#===============================================================================
echo ""
echo "[Step 6/6] Final verification..."
echo ""

conda run -n "$ENV_NAME" python << 'EOF'
import sys

print("=" * 60)
print(" Installation Verification")
print("=" * 60)
print()

# Core packages
packages = [
    ('torch', 'PyTorch'),
    ('torch_geometric', 'PyG'),
    ('torch_scatter', 'torch_scatter'),
    ('torch_sparse', 'torch_sparse'),
    ('torch_cluster', 'torch_cluster'),
    ('pyg_lib', 'pyg_lib'),
    ('numpy', 'NumPy'),
    ('scipy', 'SciPy'),
    ('pandas', 'Pandas'),
    ('sklearn', 'Scikit-learn'),
    ('optuna', 'Optuna'),
]

all_ok = True
for module_name, display_name in packages:
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'OK')
        print(f"  [OK] {display_name}: {version}")
    except ImportError as e:
        print(f"  [FAILED] {display_name}: NOT INSTALLED")
        all_ok = False

print()
print("-" * 60)

# CUDA test
import torch
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # GPU compute test
    try:
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.mm(x, x)
        del x, y
        torch.cuda.empty_cache()
        print(f"  GPU compute: PASSED")
    except Exception as e:
        print(f"  GPU compute: FAILED ({e})")

print()
print("=" * 60)
if all_ok:
    print(" SUCCESS: All packages installed correctly!")
else:
    print(" WARNING: Some packages failed to install")
print("=" * 60)
EOF

#===============================================================================
# Done
#===============================================================================
echo ""
echo "==============================================================================="
echo " Setup Complete!"
echo "==============================================================================="
echo ""
echo " To activate the environment:"
echo ""
echo "     conda activate $ENV_NAME"
echo ""
echo " Or use in scripts:"
echo ""
echo "     conda run -n $ENV_NAME python your_script.py"
echo ""
echo "==============================================================================="