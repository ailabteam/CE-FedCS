"""
Environment Check Script for the RIVF 2025 CE-FedCS Project (PyTorch version).

This script verifies the installation and basic functionality of all required libraries:
- PyTorch (and GPU detection)
- Torchvision
- Flower (flwr)
- NumPy
- Matplotlib
"""
import sys
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

def check_pytorch_environment():
    """Runs a series of checks for the required PyTorch-based libraries."""
    all_checks_passed = True
    
    print("-" * 50)
    print("🚀 Starting Environment Check for CE-FedCS (PyTorch)...")
    print("-" * 50)

    # 1. Check PyTorch and GPU
    print("\n[1] Checking PyTorch...")
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        # Check for GPU
        is_cuda_available = torch.cuda.is_available()
        if is_cuda_available:
            gpu_count = torch.cuda.device_count()
            print(f"✅ GPU DETECTED! Found {gpu_count} CUDA-enabled GPU(s):")
            for i in range(gpu_count):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠️ WARNING: No GPU detected by PyTorch.")
            print("   (The code will run on CPU, which will be much slower for CNN training.)")
            
    except ImportError:
        print("❌ CRITICAL: PyTorch is NOT installed.")
        print("   Please install it from the official website: https://pytorch.org/get-started/locally/")
        all_checks_passed = False
    except Exception as e:
        print(f"❌ An error occurred with PyTorch: {e}")
        all_checks_passed = False

    # 2. Check Torchvision
    print("\n[2] Checking Torchvision...")
    try:
        import torchvision
        print(f"✅ Torchvision version: {torchvision.__version__}")
    except ImportError:
        print("❌ CRITICAL: Torchvision is NOT installed.")
        print("   Please install it along with PyTorch.")
        all_checks_passed = False
    except Exception as e:
        print(f"❌ An error occurred with Torchvision: {e}")
        all_checks_passed = False

    # 3. Check Flower (flwr)
    print("\n[3] Checking Flower (flwr)...")
    try:
        import flwr
        print(f"✅ Flower (flwr) version: {flwr.__version__}")
    except ImportError:
        print("❌ CRITICAL: Flower (flwr) is NOT installed.")
        print("   Please install it using: pip install flwr['simulation']")
        all_checks_passed = False
    except Exception as e:
        print(f"❌ An error occurred with Flower: {e}")
        all_checks_passed = False
        
    # 4. Check NumPy & Matplotlib (remain the same)
    print("\n[4] Checking NumPy...")
    try:
        import numpy as np
        print(f"✅ NumPy version: {np.__version__}")
    except ImportError:
        print("❌ CRITICAL: NumPy is NOT installed.")
        all_checks_passed = False

    print("\n[5] Checking Matplotlib...")
    try:
        import matplotlib
        print(f"✅ Matplotlib version: {matplotlib.__version__}")
    except ImportError:
        print("❌ CRITICAL: Matplotlib is NOT installed.")
        all_checks_passed = False

    # Final Summary
    print("-" * 50)
    if all_checks_passed:
        print("🎉 SUCCESS! Your PyTorch environment is set up correctly.")
        print("   You are ready to start the experiments with Flower.")
    else:
        print("🔥 ACTION REQUIRED: Some critical packages are missing or misconfigured.")
        print("   Please review the errors above and install/fix the necessary packages.")
    print("-" * 50)

if __name__ == '__main__':
    check_pytorch_environment()
