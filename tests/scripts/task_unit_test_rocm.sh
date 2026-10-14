#!/bin/bash
# Copyright Advanced Micro Devices, Inc.
# Licensed under the Apache License Version 2.0

function fail {
    echo FAIL: $@
    exit -1
}

function usage {
    echo "Usage: $0 backend device [coverage (on or off)]"
}

if [ $# -ne 2 ] && [ $# -ne 3 ]; then
    usage
    fail "Error: must specify backend and device and optionally coverage"
fi

export DGLBACKEND=$1
export DGLTESTDEV=$2
export COVERAGE=${3:-"off"}
export DGL_LIBRARY_PATH=${PWD}/build
export PYTHONPATH=tests:${PWD}/python:$PYTHONPATH
export DGL_DOWNLOAD_DIR=${PWD}/_download
export TF_FORCE_GPU_ALLOW_GROWTH=true
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE

# torch_shm_manager, which PyTorch launches to share CPU tensors across
# processes (e.g. torch.multiprocessing, DGL's multiprocess/IPC tests), links
# against librocm-openblas.so.0 and librocprofiler-sdk.so.1. When PyTorch is
# installed from the rocm-sdk wheels those libraries live inside the package
# rather than on the default loader path: librocm-openblas.so.0 under
# .../lib/host-math/lib, librocprofiler-sdk.so.1 directly under .../lib.
# Without both on LD_LIBRARY_PATH, torch_shm_manager fails to start with
# "error while loading shared libraries", surfacing as a generic
# "no response from torch_shm_manager" RuntimeError in any test that shares
# CPU tensors across processes.
ROCM_SDK_CORE_LIB=$(python3 -c "import os, _rocm_sdk_core; print(os.path.join(os.path.dirname(_rocm_sdk_core.__file__), 'lib'))" 2>/dev/null)
if [ -d "${ROCM_SDK_CORE_LIB}" ]; then
  export LD_LIBRARY_PATH=${ROCM_SDK_CORE_LIB}:${LD_LIBRARY_PATH}
fi
ROCM_SDK_MATH_LIBS=$(python3 -c "import os, _rocm_sdk_core; print(os.path.join(os.path.dirname(_rocm_sdk_core.__file__), 'lib', 'host-math', 'lib'))" 2>/dev/null)
if [ -d "${ROCM_SDK_MATH_LIBS}" ]; then
  export LD_LIBRARY_PATH=${ROCM_SDK_MATH_LIBS}:${LD_LIBRARY_PATH}
fi

# The distributed tests rendezvous on DGL_TEST_MASTER_PORT and that port plus
# two. Their defaults are often claimed by unrelated daemons when the tests run
# in a container sharing a busy host's network namespace, so pick a free pair.
if [ -z "${DGL_TEST_MASTER_PORT}" ]; then
  export DGL_TEST_MASTER_PORT=$(python3 - <<'PY'
import socket

def available(base):
    held = []
    try:
        for port in (base, base + 2):
            sock = socket.socket()
            held.append(sock)
            sock.bind(("", port))
    except OSError:
        return False
    finally:
        for sock in held:
            sock.close()
    return True

print(next((base for base in range(12345, 12645, 4) if available(base)), 12345))
PY
)
fi
echo "distributed tests rendezvous on port ${DGL_TEST_MASTER_PORT}"

if [ $2 == "gpu" ] 
then
  export CUDA_VISIBLE_DEVICES=0
else
  export CUDA_VISIBLE_DEVICES=-1
fi

echo "pytests running without Logger"

python3 -m pip install expecttest

# These files build DataLoaders with worker processes. Forking a pytest process
# that has already accumulated thread pools from earlier tests leaves the
# workers deadlocked, so each file gets a fresh interpreter and is excluded from
# the main backend run.
ISOLATED_TESTS=""
IGNORE_ISOLATED=""
for test_file in tests/python/$DGLBACKEND/graphbolt/test_dataloader.py \
                 tests/python/$DGLBACKEND/dataloading/test_dataloader.py; do
  if [ -f ${test_file} ]; then
    ISOLATED_TESTS="${ISOLATED_TESTS} ${test_file}"
    IGNORE_ISOLATED="${IGNORE_ISOLATED} --ignore=${test_file}"
  fi
done

exit_code=0

if [ ${COVERAGE} == "off" ]; then
  echo "pytests running without coverage"
  python3 -m pytest --junitxml=pytest_dgl_import.xml --durations=100 --disable-warnings tests/python/test_dgl_import.py || exit_code=$?
  python3 -m pytest --junitxml=pytest_common.xml  --durations=100 --disable-warnings tests/python/common || exit_code=$?
  python3 -m pytest --junitxml=pytest_backend.xml --durations=100 --disable-warnings tests/python/$DGLBACKEND ${IGNORE_ISOLATED} || exit_code=$?

  for test_file in ${ISOLATED_TESTS}; do
    report_name=$(echo ${test_file} | tr '/' '_' | sed 's/\.py$//')
    python3 -m pytest --junitxml=pytest_${report_name}.xml --durations=100 --disable-warnings ${test_file} || exit_code=$?
  done

elif [ ${COVERAGE} == "on" ]; then
  echo "pytests running with coverage"
  python3 -m pip install pytest-cov
  python3 -m pytest --cov=dgl              --cov-report=lcov:lcov_pytest_import.info  --disable-warnings tests/python/test_dgl_import.py || exit_code=$?
  python3 -m pytest --cov=dgl --cov-append --cov-report=lcov:lcov_pytest_common.info  --disable-warnings tests/python/common || exit_code=$?
  python3 -m pytest --cov=dgl --cov-append --cov-report=lcov:lcov_pytest_backend.info --disable-warnings tests/python/$DGLBACKEND ${IGNORE_ISOLATED} || exit_code=$?

  isolated_traces=""
  index=0
  for test_file in ${ISOLATED_TESTS}; do
    index=$((index + 1))
    python3 -m pytest --cov=dgl --cov-append --cov-report=lcov:lcov_pytest_isolated_${index}.info --disable-warnings ${test_file} || exit_code=$?
    isolated_traces="${isolated_traces} -a lcov_pytest_isolated_${index}.info"
  done

  # TODO need to add docs for installing lcov
  lcov --add-tracefile lcov_pytest_backend.info -a lcov_pytest_common.info -a lcov_pytest_import.info ${isolated_traces} -o lcov_pytest.info

  # Show summary of coverage
  coverage report -m | tee python_coverage_report.txt

else
  fail "Error: invalid coverage option: ${COVERAGE}"
fi

echo "pytest exited with code: $exit_code"

exit $exit_code
