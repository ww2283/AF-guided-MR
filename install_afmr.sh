#!/bin/bash
# filepath: install_afmr.sh

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Always use conda for environment creation
if ! command_exists conda; then
    echo "Conda not found. Please install conda first."
    exit 1
fi

# Check if mamba exists for later use
HAS_MAMBA=0
if command_exists mamba; then
    HAS_MAMBA=1
fi

# Prompt for environment name
read -p "Enter environment name (default: automatemr): " ENV_NAME
ENV_NAME=${ENV_NAME:-automatemr}

# Add installation path prompt near start of script
read -p "Enter installation directory (default: $HOME/AF-guided-MR): " INSTALL_DIR
INSTALL_DIR=${INSTALL_DIR:-$HOME/AF-guided-MR}
INSTALL_DIR=$(realpath -m "$INSTALL_DIR")

# Create installation directory if it doesn't exist
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Check if environment exists
if conda env list | grep -q "^$ENV_NAME "; then
    read -p "Environment $ENV_NAME already exists. Remove it? (y/n): " REMOVE_ENV
    if [[ $REMOVE_ENV =~ ^[Yy]$ ]]; then
        conda env remove -n $ENV_NAME
    else
        echo "Installation aborted."
        exit 1
    fi
fi

# Detect CUDA version
if command_exists nvidia-smi; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1)
    if [[ -z "$CUDA_VERSION" ]] && command_exists nvcc; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d'.' -f1)
    fi
    
    if [[ "${CUDA_VERSION}" =~ ^[0-9]+$ ]]; then
        if [ "${CUDA_VERSION}" -ge 12 ]; then
            JAX_CUDA="cuda12_pip"
        else
            JAX_CUDA="cuda11_pip"
        fi
    else
        echo "Could not determine CUDA version. Defaulting to CUDA 11"
        JAX_CUDA="cuda11_pip"
    fi
else
    echo "NVIDIA drivers not found. Please install CUDA first."
    exit 1
fi

# Create environment using conda
echo "Creating new environment $ENV_NAME..."
conda create -n $ENV_NAME python=3.10 -y

# Source conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Verify activation
if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    echo "Failed to activate environment $ENV_NAME"
    exit 1
fi

# Use mamba if available for faster package installation
PKG_MGR="conda"
if [ $HAS_MAMBA -eq 1 ]; then
    PKG_MGR="mamba"
fi

# Install dependencies
$PKG_MGR install -y -c conda-forge -c bioconda openmm==7.7.0 pdbfixer kalign2=2.04 hhsuite=3.3.0 cctbx-base

# Install ColabFold
pip install --no-warn-conflicts "colabfold[alphafold-without-jax] @ git+https://github.com/sokrypton/ColabFold"
pip install --upgrade "jax[${JAX_CUDA}]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install "colabfold[alphafold]"

# Fix potential pdbfixer issue
CONDA_BASE=$(conda info --base)
PDBFIXER_PATH=$(find "${CONDA_BASE}/envs/$ENV_NAME" -name pdbfixer.py)
if [ -f "$PDBFIXER_PATH" ]; then
    sed -i 's/simtk.openmm/openmm/g' "$PDBFIXER_PATH"
fi

# Clone repository and install remaining dependencies
if [ ! -d "$INSTALL_DIR/AF-guided-MR" ]; then
    git clone https://github.com/ww2283/AF-guided-MR "$INSTALL_DIR/AF-guided-MR"
fi
pip install nvidia-ml-py3 gemmi mdtraj polyleven pandarallel scikit-learn hdbscan colorama biopython psutil pycuda

# Define alias command
ALIAS_CMD="alias mr='conda run -n ${ENV_NAME} python ${INSTALL_DIR}/AF-guided-MR/main.py'"

# Check if exact alias exists
if ! grep -Fxq "${ALIAS_CMD}" ~/.bash_aliases 2>/dev/null; then
    echo "${ALIAS_CMD}" >> ~/.bash_aliases
fi

echo "Installation complete!"
echo "AF-guided-MR installed to: $INSTALL_DIR"
echo "Please restart your terminal or run: source ~/.bash_aliases"