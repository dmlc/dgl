#!/bin/bash
#
# Build DGL with Huawei Ascend NPU support
#
# Usage:
#   bash script/build_dgl_ascend.sh [options]
#
# Options:
#   -s SOC     Ascend SoC version (auto-detected by default)
#   -t TYPE    Build type: dev, dogfood, release (default: dev)
#   -c         Clean build (remove existing build directory)
#   -j JOBS    Number of parallel jobs (default: $(nproc))
#   -h         Show this help message
#
# Environment variables:
#   ASCEND_TOOLKIT_HOME: Path to Ascend CANN toolkit (optional, auto-detected)
#   DGL_HOME: Path to DGL source directory (optional, auto-detected)
#   SOC_VERSION: Override SoC version (same as -s)
#
# Supported SoC versions:
#   A2 series:  Ascend910B1, Ascend910B2, Ascend910B2C, Ascend910B3, Ascend910B4
#   A3 series:  ascend910_9362, ascend910_9372, ascend910_9381, ascend910_9382,
#               ascend910_9391, ascend910_9392
#   A5 series:  ascend950pr_9579
# Note: A2 and A3 share the same DAV_2201 architecture (same instruction set,
#   same buffer sizes). The only differences are core count (20/40 vs 24/48),
#   frequency, L2 and HBM size. Core count is queried at runtime via
#   aclrtGetDeviceInfo, so one build runs on all sub-models. SOC_VERSION only
#   affects the compiled kernel microcode binary.
#
# A5 (Ascend 950PR, DAV_3510) is a separate architecture: build with the
#   auto-detected SOC_VERSION (kernels are source-compatible; UB is larger,
#   248KB vs 192KB, and core counts are queried at runtime).
set -e
set -o pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo_info()  { echo -e "${GREEN}[INFO]${NC} $1" >&2; }
echo_warn()  { echo -e "${YELLOW}[WARN]${NC} $1" >&2; }
echo_error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${DGL_HOME}" ]]; then
    export DGL_HOME="${PROJECT_ROOT}"
    echo_info "DGL_HOME set to: ${DGL_HOME}"
fi
cd "${DGL_HOME}"
echo_info "Working directory: $(pwd)"

# ---------------------------------------------------------------------------
# Supported SoC version lists
# ---------------------------------------------------------------------------
A2_SOC_LIST="ascend910b1 ascend910b2 ascend910b2c ascend910b3 ascend910b4 ascend910b4-1"
A3_SOC_LIST="ascend910_9362 ascend910_9372 ascend910_9381 ascend910_9382 ascend910_9391 ascend910_9392"
# A5 (Ascend 950PR, DAV_3510): only explicitly verified models; the model suffix
# comes from aclrtGetSocName / npu-smi "NPU Name" (e.g. 9579 -> Ascend950PR_9579)
A5_SOC_LIST="ascend950pr_9579"
ALL_SOC_LIST="${A2_SOC_LIST} ${A3_SOC_LIST} ${A5_SOC_LIST}"

is_valid_soc() {
    local soc_lower
    soc_lower=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    for s in ${ALL_SOC_LIST}; do
        [[ "${soc_lower}" == "${s}" ]] && return 0
    done
    return 1
}

is_a5_soc() {
    local soc_lower
    soc_lower=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    for s in ${A5_SOC_LIST}; do
        [[ "${soc_lower}" == "${s}" ]] && return 0
    done
    return 1
}

is_a2_soc() {
    local soc_lower
    soc_lower=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    for s in ${A2_SOC_LIST}; do
        [[ "${soc_lower}" == "${s}" ]] && return 0
    done
    return 1
}

# ---------------------------------------------------------------------------
# Auto-detect SoC version from hardware
#   Strategy: npu-smi board info → NPU Name field → map to SoC version
#   Fallback: ascend910b4 (A2, also runs on A3; DAV_2201)
# ---------------------------------------------------------------------------
detect_soc_version() {
    # Strategy 1: exact SoC string from the driver via aclrtGetSocName.
    # Works on all series (incl. Ascend 950PR, e.g. "Ascend950PR_9579") and the
    # returned string is accepted by the AscendC compiler verbatim.
    # NOTE: needs CANN env, so detect_ascend_toolkit + setup_ascend_env run
    # before SoC resolution (see main flow below).
    local soc_name
    soc_name=$(python3 -c "
import ctypes
try:
    lib = ctypes.CDLL('libascendcl.so')
    lib.aclrtGetSocName.restype = ctypes.c_char_p
    print((lib.aclrtGetSocName() or b'').decode())
except Exception:
    pass
" 2>/dev/null || true)
    if [[ -n "${soc_name}" ]]; then
        echo_info "Detected SoC via aclrtGetSocName: ${soc_name}"
        echo "${soc_name}"
        return
    fi

    if ! command -v npu-smi &>/dev/null; then
        echo_warn "npu-smi not found, cannot auto-detect SoC version"
        echo "ascend910b4"
        return
    fi

    # Strategy 2: npu-smi board info. Some boards (e.g. A310-50) do not
    # support "-c 0", so retry without it.
    local npu_name
    npu_name=$(npu-smi info -t board -i 0 -c 0 2>/dev/null | grep "NPU Name" | awk -F':' '{print $2}' | tr -d ' \t' || true)
    if [[ -z "${npu_name}" ]]; then
        npu_name=$(npu-smi info -t board -i 0 2>/dev/null | grep "NPU Name" | awk -F':' '{print $2}' | tr -d ' \t' || true)
    fi

    if [[ -z "${npu_name}" ]]; then
        echo_warn "Cannot read NPU Name from npu-smi, using default ascend910b4"
        echo "ascend910b4"
        return
    fi

    echo_info "Detected NPU Name: ${npu_name}"

    # Map board name to SoC version
    case "${npu_name}" in
        910B1|910b1)       echo "ascend910b1" ;;
        910B2|910b2)       echo "ascend910b2" ;;
        910B2C|910b2c)     echo "ascend910b2c" ;;
        910B3|910b3)       echo "ascend910b3" ;;
        910B4|910b4)       echo "ascend910b4" ;;
        9362)              echo "ascend910_9362" ;;
        9372)              echo "ascend910_9372" ;;
        9381)              echo "ascend910_9381" ;;
        9382)              echo "ascend910_9382" ;;
        9391)              echo "ascend910_9391" ;;
        9392)              echo "ascend910_9392" ;;
        9579)              echo "Ascend950PR_9579" ;;
        *)
            echo_warn "Unknown NPU Name '${npu_name}', using default ascend910b4"
            echo "ascend910b4"
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Detect Ascend CANN toolkit
# ---------------------------------------------------------------------------
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

setup_ascend_env() {
    if [[ -n "${ASCEND_TOOLKIT_HOME}" ]]; then
        local env_script="${ASCEND_TOOLKIT_HOME}/bin/setenv.bash"
        if [[ -f "${env_script}" ]]; then
            echo_info "Sourcing Ascend environment: ${env_script}"
            set +e
            source "${env_script}" 2>/dev/null || true
            set -e
        fi
        export LD_LIBRARY_PATH="${ASCEND_TOOLKIT_HOME}/lib64:${LD_LIBRARY_PATH}"
        export PATH="${ASCEND_TOOLKIT_HOME}/bin:${PATH}"
        echo_info "LD_LIBRARY_PATH updated"
    fi
}

# ---------------------------------------------------------------------------
# Parse command line arguments
# ---------------------------------------------------------------------------
BUILD_TYPE="dev"
CLEAN_BUILD=false
JOBS=$(nproc)
SOC_VERSION=""

while getopts "hs:t:cj:" opt; do
    case ${opt} in
        h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -s SOC     Ascend SoC version (auto-detected by default)"
            echo "             A2: $(echo ${A2_SOC_LIST} | sed 's/ /, /g')"
            echo "             A3: $(echo ${A3_SOC_LIST} | sed 's/ /, /g')
             A5: $(echo ${A5_SOC_LIST} | sed 's/ /, /g') (auto-detected via aclrtGetSocName)"
            echo "  -t TYPE    Build type: dev, dogfood, release (default: dev)"
            echo "  -c         Clean build (remove existing build directory)"
            echo "  -j JOBS    Number of parallel jobs (default: $(nproc))"
            echo "  -h         Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  SOC_VERSION          Override SoC version (same as -s)"
            echo "  ASCEND_TOOLKIT_HOME  Path to CANN toolkit"
            echo "  DGL_HOME             Path to DGL source"
            exit 0
            ;;
        s)
            SOC_VERSION="${OPTARG}"
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

# Detect and setup Ascend environment first: SoC auto-detection via
# aclrtGetSocName needs libascendcl.so on the library path.
detect_ascend_toolkit || true
setup_ascend_env

# Resolve SoC version: CLI -s > env SOC_VERSION > auto-detect
if [[ -z "${SOC_VERSION}" ]]; then
    SOC_VERSION=$(detect_soc_version)
fi

# Strip whitespace
SOC_VERSION="$(echo "${SOC_VERSION}" | tr -d '[:space:]')"

# Validate
if ! is_valid_soc "${SOC_VERSION}"; then
    echo_error "Invalid SoC version: ${SOC_VERSION}"
    echo_error "Supported: ${ALL_SOC_LIST}"
    exit 1
fi

if is_a5_soc "${SOC_VERSION}"; then
    echo_info "SoC: ${SOC_VERSION} (Atlas A5 series, DAV_3510)"
elif is_a2_soc "${SOC_VERSION}"; then
    echo_info "SoC: ${SOC_VERSION} (Atlas A2 series, DAV_2201)"
else
    echo_info "SoC: ${SOC_VERSION} (Atlas A3 series, DAV_2201)"
fi

echo_info "=========================================="
echo_info "  DGL Ascend Build Script"
echo_info "=========================================="
echo_info "Build type: ${BUILD_TYPE}"
echo_info "SoC version: ${SOC_VERSION}"
echo_info "Parallel jobs: ${JOBS}"

# Clean build if requested
if [[ "${CLEAN_BUILD}" == true ]]; then
    echo_info "Cleaning previous build..."
    rm -rf build
    rm -rf graphbolt/build
    rm -rf dgl_sparse/build
    rm -rf tensoradapter/pytorch/build
else
    # Incremental build: clean AscendC kernel preprocess subdirectories.
    # merge_obj_text.sh links .o files in-place (REL→EXEC), so re-running
    # the build on an existing directory causes ld.lld "unknown file type".
    if [[ -d "build" ]]; then
        echo_info "Cleaning AscendC kernel preprocess cache for incremental build..."
        find build -maxdepth 1 -type d -name "dgl_ascendc_*_preprocess*" -exec rm -rf {} + 2>/dev/null || true
    fi
fi

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
    -DSOC_VERSION="${SOC_VERSION}"
)

if [[ -n "${ASCEND_TOOLKIT_HOME}" ]]; then
    CMAKE_ARGS+=(-DASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME}")
fi

TORCH_CMAKE_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null) || true
if [[ -n "${TORCH_CMAKE_PATH}" ]]; then
    CMAKE_ARGS+=(-DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:+${CMAKE_PREFIX_PATH};}${TORCH_CMAKE_PATH}")
    echo_info "PyTorch cmake path: ${TORCH_CMAKE_PATH}"
fi

TORCH_NPU_LIB_PATH=$(python3 -c "import torch_npu; import os; print(os.path.join(os.path.dirname(torch_npu.__file__), 'lib'))" 2>/dev/null) || true
if [[ -n "${TORCH_NPU_LIB_PATH}" ]] && [[ -d "${TORCH_NPU_LIB_PATH}" ]]; then
    # libgfortran.so / libgomp.so live in the GCC-private directory which is
    # not on the default find_library search path; merge it into
    # CMAKE_LIBRARY_PATH so the dgl link can resolve them.
    GCC_LIB_DIR=""
    for d in /usr/lib/gcc/aarch64-linux-gnu/* /usr/lib/gcc/aarch64-linux-gnu/*/*; do
        if [[ -f "${d}/libgfortran.so" ]]; then GCC_LIB_DIR="${d}"; break; fi
    done
    if [[ -z "${GCC_LIB_DIR}" ]] && [[ -d "/usr/lib/gcc/aarch64-linux-gnu/10.3.1" ]]; then
        GCC_LIB_DIR="/usr/lib/gcc/aarch64-linux-gnu/10.3.1"
    fi
    if [[ -n "${GCC_LIB_DIR}" ]] && [[ -d "${GCC_LIB_DIR}" ]]; then
        echo_info "Adding GCC lib dir to CMAKE_LIBRARY_PATH: ${GCC_LIB_DIR}"
        CMAKE_ARGS+=(-DCMAKE_LIBRARY_PATH="${TORCH_NPU_LIB_PATH};${GCC_LIB_DIR}")
    else
        CMAKE_ARGS+=(-DCMAKE_LIBRARY_PATH="${TORCH_NPU_LIB_PATH}")
    fi
    export LIBRARY_PATH="${LIBRARY_PATH:+${LIBRARY_PATH}:}${TORCH_NPU_LIB_PATH}"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${TORCH_NPU_LIB_PATH}"
    echo_info "torch_npu lib path: ${TORCH_NPU_LIB_PATH}"
fi

echo_info "CMake arguments: ${CMAKE_ARGS[*]}"
cmake "${CMAKE_ARGS[@]}" ..

# Build
echo_info "Building DGL with ${JOBS} parallel jobs..."
make -j${JOBS}

if [[ $? -eq 0 ]]; then
    echo_info "=========================================="
    echo_info "  Build completed successfully!"
    echo_info "=========================================="
    echo ""
    echo_info "SoC version: ${SOC_VERSION}"
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
