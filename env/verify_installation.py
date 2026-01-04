#!/usr/bin/env python3
"""
Verification script for PyTorch + PyG installation.
Checks all installed packages and CUDA availability.
"""

import sys

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_status(name, status, details=""):
    symbol = "✓" if status else "✗"
    color_code = "\033[92m" if status else "\033[91m"
    reset_code = "\033[0m"
    detail_str = f" ({details})" if details else ""
    print(f"  {color_code}[{symbol}]{reset_code} {name}{detail_str}")

def check_torch():
    print_header("PyTorch Installation")
    try:
        import torch
        print_status("torch", True, f"version {torch.__version__}")
        print_status("CUDA available", torch.cuda.is_available())
        
        if torch.cuda.is_available():
            print_status("CUDA version", True, torch.version.cuda)
            print_status("cuDNN version", True, str(torch.backends.cudnn.version()))
            print_status("GPU count", True, str(torch.cuda.device_count()))
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print_status(f"GPU {i}", True, f"{props.name}, {props.total_memory / 1024**3:.1f} GB")
        
        return True
    except ImportError as e:
        print_status("torch", False, str(e))
        return False

def check_torchvision():
    try:
        import torchvision
        print_status("torchvision", True, f"version {torchvision.__version__}")
        return True
    except ImportError as e:
        print_status("torchvision", False, str(e))
        return False

def check_torchaudio():
    try:
        import torchaudio
        print_status("torchaudio", True, f"version {torchaudio.__version__}")
        return True
    except ImportError as e:
        print_status("torchaudio", False, str(e))
        return False

def check_pyg():
    print_header("PyG Installation")
    all_ok = True
    
    try:
        import torch_geometric
        print_status("torch-geometric", True, f"version {torch_geometric.__version__}")
    except ImportError as e:
        print_status("torch-geometric", False, str(e))
        all_ok = False
    
    try:
        import torch_scatter
        print_status("torch-scatter", True)
    except ImportError as e:
        print_status("torch-scatter", False, str(e))
        all_ok = False
    
    try:
        import torch_sparse
        print_status("torch-sparse", True)
    except ImportError as e:
        print_status("torch-sparse", False, str(e))
        all_ok = False
    
    try:
        import torch_cluster
        print_status("torch-cluster", True)
    except ImportError as e:
        print_status("torch-cluster", False, str(e))
        all_ok = False
    
    try:
        import torch_spline_conv
        print_status("torch-spline-conv", True)
    except ImportError as e:
        print_status("torch-spline-conv", False, str(e))
        all_ok = False
    
    try:
        import pyg_lib
        print_status("pyg-lib", True)
    except ImportError as e:
        print_status("pyg-lib", False, str(e))
        all_ok = False
    
    return all_ok

def check_scientific():
    print_header("Scientific Computing Packages")
    all_ok = True
    
    packages = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
    ]
    
    for name, module in packages:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "unknown")
            print_status(name, True, f"version {version}")
        except ImportError as e:
            print_status(name, False, str(e))
            all_ok = False
    
    return all_ok

def check_utilities():
    print_header("Utility Packages")
    all_ok = True
    
    packages = [
        ("jupyterlab", "jupyterlab"),
        ("tqdm", "tqdm"),
        ("rich", "rich"),
        ("networkx", "networkx"),
        ("optuna", "optuna"),
    ]
    
    for name, module in packages:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "unknown")
            print_status(name, True, f"version {version}")
        except ImportError as e:
            print_status(name, False, str(e))
            all_ok = False
    
    return all_ok

def run_gpu_test():
    print_header("GPU Functionality Test")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print_status("GPU test", False, "CUDA not available")
            return False
        
        # Simple tensor operation on GPU
        device = torch.device("cuda:0")
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.matmul(x, y)
        
        print_status("GPU tensor operation", True, f"result shape: {z.shape}")
        
        # PyG operation test
        from torch_geometric.data import Data
        from torch_geometric.nn import GCNConv
        
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long, device=device)
        x = torch.randn(3, 16, device=device)
        
        conv = GCNConv(16, 32).to(device)
        out = conv(x, edge_index)
        
        print_status("PyG GCN on GPU", True, f"output shape: {out.shape}")
        
        return True
        
    except Exception as e:
        print_status("GPU test", False, str(e))
        return False

def main():
    print("\n" + "="*60)
    print(" PyTorch + PyG Installation Verification")
    print("="*60)
    
    results = []
    
    results.append(("PyTorch", check_torch()))
    check_torchvision()
    check_torchaudio()
    results.append(("PyG", check_pyg()))
    results.append(("Scientific", check_scientific()))
    results.append(("Utilities", check_utilities()))
    results.append(("GPU Test", run_gpu_test()))
    
    print_header("Summary")
    
    all_passed = True
    for name, passed in results:
        print_status(name, passed)
        if not passed:
            all_passed = False
    
    print()
    
    if all_passed:
        print("\033[92m" + "All checks passed! Environment is ready." + "\033[0m")
        return 0
    else:
        print("\033[91m" + "Some checks failed. Please review the output above." + "\033[0m")
        return 1

if __name__ == "__main__":
    sys.exit(main())