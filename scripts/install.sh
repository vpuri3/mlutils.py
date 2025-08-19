#!/bin/bash
#=========================================#
# # Eagle
# TORCH_VERSION=2.6
# CUDA_VERSION=cu124
#=========================================#
# # GCloud H100
# TORCH_VERSION=2.7
# CUDA_VERSION=cu128
#=========================================#
# Orchard
TORCH_VERSION=2.6
CUDA_VERSION=cu124

#=========================================#
# Update uv
#=========================================#
echo "Updating uv..."
uv self update

set -e  # Exit on any error

#=========================================#
# Create virtual environment
#=========================================#
echo "#------------------------#"
echo "Cleaning up previous installation..."
echo "#------------------------#"
rm -rf .venv uv.lock main.py pyproject.toml

echo "#------------------------#"
echo "Initializing project..."
echo "#------------------------#"
uv init --name flare --python=3.11 --no-readme

echo "#------------------------#"
echo "Creating virtual environment..."
echo "#------------------------#"
uv venv

#=========================================#
# Install packages
#=========================================#

# Install PyTorch first directly into virtual environment to avoid dependency conflicts
echo "#------------------------#"
echo "Installing PyTorch with CUDA support..."
echo "#------------------------#"
uv pip install torch==${TORCH_VERSION} torchvision --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

# Now add other packages from PyPI (this will use default PyPI index)
echo "#------------------------#"
echo "Installing core packages from PyPI..."
echo "#------------------------#"
uv pip install torch_geometric
uv add timm datasets
uv add tqdm jsonargparse einops setuptools packaging
uv add scipy pandas seaborn pyvista matplotlib

# Interactive tools
echo "#------------------------#"
echo "Installing interactive tools..."
echo "#------------------------#"
uv add ipython gpustat

# Flash Attention with proper build configuration
read -p "Install Flash Attention for faster transformer models? [y/N] " install_flash_attn
if [[ $install_flash_attn == [Yy]* ]]; then
    echo "Installing Flash Attention..."
    uv add flash-attn --no-build-isolation
    echo "Flash Attention installation completed."
else
    echo "Skipping Flash Attention installation."
fi

# LaTeX for high-quality plots (requires sudo access)
read -p "Install LaTeX for publication-quality plots? (requires sudo access) [y/N] " install_latex
if [[ $install_latex == [Yy]* ]]; then
    echo "Installing LaTeX packages..."
    sudo apt update && sudo apt install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended
    sudo apt install -y texlive-fonts-extra texlive-latex-extra cm-super
    sudo apt install -y dvipng
    echo "LaTeX installation completed."
else
    echo "Skipping LaTeX installation."
fi

echo "#------------------------#"
echo "Installation completed successfully!"
echo "Activate the environment with: source .venv/bin/activate"
echo "#------------------------#"
#