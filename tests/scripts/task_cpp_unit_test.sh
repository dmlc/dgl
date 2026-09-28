#!/bin/bash
function fail {
    echo FAIL: $@
    exit -1
}

# Resolve the repo root relative to this script's own location, so this
# works regardless of the caller's current directory. (Previously this
# assumed the caller was already at the repo root; running it from inside
# build/ made `pushd build` fail with "No such file or directory".)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}" || fail "cannot cd to repo root (${REPO_ROOT})"

echo $PWD
pushd build
ls -lh

# ROCm 10.0+ ships via pip across several packages (rocm-sdk-core,
# rocm-sdk-libraries, rocm-sdk-device-*), each with its own runtime .so files
# living inside that package's site-packages dir rather than on the default
# loader path. Without this, runUnitTests fails one missing lib at a time as
# the loader walks the NEEDED list (e.g. "libamdhip64.so.7", then
# "libhipsparse.so.4", then potentially libhipblas/librocsparse/libMIOpen...).
# Rather than chasing those individually, discover every directory across all
# rocm_sdk_* packages that actually contains a .so and add them all at once.
# No-op on non-rocm-sdk builds (classic /opt/rocm FHS installs, CUDA,
# CPU-only), since the import silently fails and ROCM_SDK_CORE_DIR stays empty.
ROCM_SDK_CORE_DIR=$(python3 -c "import os, _rocm_sdk_core; print(os.path.dirname(os.path.dirname(_rocm_sdk_core.__file__)))" 2>/dev/null)
if [ -d "${ROCM_SDK_CORE_DIR}" ]; then
  ROCM_SDK_LIB_DIRS=$(find "${ROCM_SDK_CORE_DIR}" -maxdepth 10 -path '*rocm_sdk*' -name '*.so*' -printf '%h\n' 2>/dev/null | sort -u | paste -sd: -)
  if [ -n "${ROCM_SDK_LIB_DIRS}" ]; then
    export LD_LIBRARY_PATH=${ROCM_SDK_LIB_DIRS}:${LD_LIBRARY_PATH}
  fi
fi

export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
./runUnitTests || fail "CPP unit test"
popd
