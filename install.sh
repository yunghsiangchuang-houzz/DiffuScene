#!/bin/bash
# Install script for DiffuScene
# Creates a local conda environment and installs all dependencies

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Installing DiffuScene"
echo "=========================================="

# Step 1: Create conda environment locally
echo ""
echo "Step 1: Creating conda environment locally..."
ENV_PATH="$SCRIPT_DIR/conda-envs/diffuscene"
if [ -d "$ENV_PATH" ]; then
    echo "Warning: Local environment directory already exists at $ENV_PATH"
    read -p "Do you want to remove it and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$ENV_PATH"
    else
        echo "Skipping environment creation. Using existing environment."
    fi
fi

if [ ! -d "$ENV_PATH" ]; then
    conda env create -f environment.yaml --prefix "$ENV_PATH"
    echo "Conda environment created at $ENV_PATH"
else
    echo "Using existing conda environment at $ENV_PATH"
fi

# Step 2: Activate conda environment
echo ""
echo "Step 2: Activating conda environment..."
# Initialize conda for bash shell
eval "$(conda shell.bash hook)"
conda activate "$ENV_PATH"

# Verify activation
if [ "$CONDA_DEFAULT_ENV" != "$ENV_PATH" ]; then
    echo "Error: Failed to activate conda environment"
    exit 1
fi

echo "Conda environment activated: $CONDA_DEFAULT_ENV"

# Step 2a: Install compatible PyTorch, torchvision, and torchtext versions
echo ""
echo "Step 2a: Installing compatible PyTorch, torchvision, and torchtext versions..."
# Install PyTorch ecosystem from PyTorch's official index to ensure compatibility
# Using PyTorch 2.2.0+cu121, torchvision, and torchtext as a compatible set
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
pip install --upgrade pip
pip install torch==2.2.0+cu121 torchvision torchtext --index-url "$PYTORCH_INDEX_URL"
echo "PyTorch, torchvision, and torchtext installed with compatible versions"

# Step 2b: Install CUDA toolkit for compiling CUDA extensions
echo ""
echo "Step 2b: Installing CUDA toolkit (nvcc compiler and headers)..."
# Get PyTorch's CUDA version and install matching toolkit
PYTORCH_CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "12.1")
CUDA_MAJOR_MINOR=$(echo $PYTORCH_CUDA_VERSION | cut -d'.' -f1,2)
echo "PyTorch CUDA version: $PYTORCH_CUDA_VERSION, installing cuda-toolkit=$CUDA_MAJOR_MINOR"
conda install -y -c nvidia cuda-toolkit=$CUDA_MAJOR_MINOR || conda install -y -c nvidia cuda-toolkit

# Fix potential broken libcudart.so symlink
if [ -L "$ENV_PATH/lib/libcudart.so" ] && [ ! -e "$ENV_PATH/lib/libcudart.so" ]; then
    echo "Fixing broken libcudart.so symlink..."
    CUDART_TARGET=$(ls "$ENV_PATH/lib/libcudart.so."* 2>/dev/null | grep -v ".so.1" | head -1)
    if [ -n "$CUDART_TARGET" ]; then
        rm "$ENV_PATH/lib/libcudart.so"
        ln -s "$(basename $CUDART_TARGET)" "$ENV_PATH/lib/libcudart.so"
        echo "Fixed: libcudart.so -> $(basename $CUDART_TARGET)"
    fi
fi

# Set CUDA environment variables
export CUDA_HOME="$ENV_PATH"
export LD_LIBRARY_PATH="$ENV_PATH/lib/python3.8/site-packages/torch/lib:$ENV_PATH/lib:$LD_LIBRARY_PATH"
echo "CUDA_HOME set to: $CUDA_HOME"

# Step 3: Compile extension modules
echo ""
echo "Step 3: Compiling extension modules..."
python setup.py build_ext --inplace

# Step 4: Install package in editable mode
echo ""
echo "Step 4: Installing package in editable mode..."
pip install -e .

# Step 4c: Install open3d (with --ignore-installed to handle PyYAML conflicts)
echo ""
echo "Step 4b: Installing open3d..."
pip install --ignore-installed pyyaml open3d
echo "open3d installed successfully"

# Step 5: Install ChamferDistancePytorch
echo ""
echo "Step 5: Installing ChamferDistancePytorch..."
cd ChamferDistancePytorch/chamfer3D
python setup.py install
cd "$SCRIPT_DIR"

echo ""
echo "=========================================="
echo "Installation completed successfully!"
echo "=========================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  conda activate $ENV_PATH"
echo ""
echo "Or use:"
echo "  source activate.sh"
echo ""
