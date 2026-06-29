#!/bin/bash
#
# Build DGL with Huawei Ascend NPU support
#
# Usage:
#   bash script/build_dgl_ascend.sh
#
# Environment variables:
#   ASCEND_TOOLKIT_HOME: Path to Ascend CANN toolkit (optional, auto-detected)
#   DGL_HOME: Path to DGL source directory (optional, auto-detected)
#

# Exit on error, but we'll handle some cases specially
set -e
set -o pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Set DGL_HOME if not set
if [[ -z "${DGL_HOME}" ]]; then
    export DGL_HOME="${PROJECT_ROOT}"
    echo_info "DGL_HOME set to: ${DGL_HOME}"
fi

# Check if we're in the correct directory
cd "${DGL_HOME}"
echo_info "Working directory: $(pwd)"

# Detect Ascend CANN toolkit
detect_ascend_toolkit() {
    if [[ -n "${ASCEND_TOOLKIT_HOME}" ]] && [[ -d "${ASCEND_TOOLKIT_HOME}" ]]; then
        echo_info "Using ASCEND_TOOLKIT_HOME: ${ASCEND_TOOLKIT_HOME}"
        return 0
    fi
    
    if [[ -n "${ASCEND_HOME}" ]] && [[ -d "${ASCEND_HOME}" ]]; then
        export ASCEND_TOOLKIT_HOME="${ASCEND_HOME}"
        echo_info "Using ASCEND_HOME: ${ASCEND_TOOLKIT_HOME}"
        return 0
    fi
    
    # Try common installation paths
    local common_paths=(
        "/usr/local/Ascend/ascend-toolkit/latest"
        "/usr/local/Ascend/latest"
        "$HOME/Ascend/ascend-toolkit/latest"
    )
    
    for path in "${common_paths[@]}"; do
        if [[ -d "${path}" ]]; then
            export ASCEND_TOOLKIT_HOME="${path}"
            echo_info "Auto-detected Ascend toolkit at: ${ASCEND_TOOLKIT_HOME}"
            return 0
        fi
    done
    
    echo_warn "Ascend CANN toolkit not found. Building without ACL support."
    echo_warn "Set ASCEND_TOOLKIT_HOME environment variable if CANN is installed elsewhere."
    return 1
}

# Setup Ascend environment
setup_ascend_env() {
    if [[ -n "${ASCEND_TOOLKIT_HOME}" ]]; then
        # Source the Ascend environment script if it exists
        local env_script="${ASCEND_TOOLKIT_HOME}/bin/setenv.bash"
        if [[ -f "${env_script}" ]]; then
            echo_info "Sourcing Ascend environment: ${env_script}"
            # Temporarily disable set -e to avoid exit from setenv.bash
            set +e
            source "${env_script}" 2>/dev/null || true
            set -e
        fi
        # Always set paths manually to ensure they are correct
        export LD_LIBRARY_PATH="${ASCEND_TOOLKIT_HOME}/lib64:${LD_LIBRARY_PATH}"
        export PATH="${ASCEND_TOOLKIT_HOME}/bin:${PATH}"
        echo_info "LD_LIBRARY_PATH updated"
    fi
}

# Parse command line arguments
BUILD_TYPE="dev"
CLEAN_BUILD=false
JOBS=$(nproc)

while getopts "ht:cj:" opt; do
    case ${opt} in
        h)
            echo "Usage: $0 [-t build_type] [-c] [-j jobs]"
            echo ""
            echo "Options:"
            echo "  -t TYPE    Build type: dev, dogfood, release (default: dev)"
            echo "  -c         Clean build (remove existing build directory)"
            echo "  -j JOBS    Number of parallel jobs (default: $(nproc))"
            echo "  -h         Show this help message"
            exit 0
            ;;
        t)
            BUILD_TYPE="${OPTARG}"
            ;;
        c)
            CLEAN_BUILD=true
            ;;
        j)
            JOBS="${OPTARG}"
            ;;
        \?)
            echo "Invalid option: -${OPTARG}" >&2
            exit 1
            ;;
    esac
done

echo_info "=========================================="
echo_info "  DGL Ascend Build Script"
echo_info "=========================================="
echo_info "Build type: ${BUILD_TYPE}"
echo_info "Parallel jobs: ${JOBS}"

# Detect and setup Ascend environment
detect_ascend_toolkit || true  # Continue even if not found
setup_ascend_env

# Clean build if requested
if [[ "${CLEAN_BUILD}" == true ]]; then
    echo_info "Cleaning previous build..."
    rm -rf build
    rm -rf graphbolt/build
    rm -rf dgl_sparse/build
    rm -rf tensoradapter/pytorch/build
fi

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo_info "Configuring with CMake..."
CMAKE_ARGS=(
    -DBUILD_TYPE="${BUILD_TYPE}"
    -DUSE_CUDA=OFF
    -DUSE_ASCEND=ON
    -DBUILD_TORCH=ON
    -DUSE_DISTRIBUTED=ON
    -DUSE_GRAPHBLAST=ON
    -DUSE_TORCH=ON
)

# Add Ascend toolkit path if available
if [[ -n "${ASCEND_TOOLKIT_HOME}" ]]; then
    CMAKE_ARGS+=(-DASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME}")
fi

# Append PyTorch cmake path if available
TORCH_CMAKE_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null) || true
if [[ -n "${TORCH_CMAKE_PATH}" ]]; then
    CMAKE_ARGS+=(-DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:+${CMAKE_PREFIX_PATH};}${TORCH_CMAKE_PATH}")
    echo_info "PyTorch cmake path: ${TORCH_CMAKE_PATH}"
fi

# Add torch_npu lib path for find_library and linker
TORCH_NPU_LIB_PATH=$(python3 -c "import torch_npu; import os; print(os.path.join(os.path.dirname(torch_npu.__file__), 'lib'))" 2>/dev/null) || true
if [[ -n "${TORCH_NPU_LIB_PATH}" ]] && [[ -d "${TORCH_NPU_LIB_PATH}" ]]; then
    CMAKE_ARGS+=(-DCMAKE_LIBRARY_PATH="${TORCH_NPU_LIB_PATH}")
    export LIBRARY_PATH="${LIBRARY_PATH:+${LIBRARY_PATH}:}${TORCH_NPU_LIB_PATH}"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${TORCH_NPU_LIB_PATH}"
    echo_info "torch_npu lib path: ${TORCH_NPU_LIB_PATH}"
fi

echo_info "CMake arguments: ${CMAKE_ARGS[*]}"
cmake "${CMAKE_ARGS[@]}" ..

# Build
echo_info "Building DGL with ${JOBS} parallel jobs..."
make -j${JOBS}

# Check if build was successful
if [[ $? -eq 0 ]]; then
    echo_info "=========================================="
    echo_info "  Build completed successfully!"
    echo_info "=========================================="
    echo ""
    echo_info "To install the Python package, run:"
    echo "  cd ${DGL_HOME}/python"
    echo "  pip install -e ."
    echo ""
    echo_info "To test the installation:"
    echo "  python -c \"import dgl; print(dgl.__version__)\""
else
    echo_error "Build failed!"
    exit 1
fi

