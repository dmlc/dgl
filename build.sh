#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS="${JOBS:-64}"
MODE="${1:-fast}"

cd "${PROJECT_ROOT}"

full_build() {
    rm -rf build
    rm -rf tensoradapter/pytorch/build
    rm -rf graphbolt/build
    rm -rf dgl_sparse/build

    bash script/build_dgl_ascend.sh -j "${JOBS}"
    python3 -m pip install --no-build-isolation -e ./python
}

case "${MODE}" in
    fast)
        if [[ ! -f build/Makefile || ! -f build/libdgl.so ]]; then
            echo "Build artifacts not found; running a full build first."
            full_build
        else
            cmake --build build --target dgl/fast -j "${JOBS}"
        fi
        ;;
    full)
        full_build
        ;;
    test)
        python3 tests/ascend/test_spmm_sum_npu.py \
            --runs 1 \
            --measurements 1 \
            --warmup 1
        ;;
    bench)
        python3 tests/ascend/test_spmm_sum_npu.py
        ;;
    *)
        echo "Usage: bash build.sh {fast|full|test|bench}" >&2
        exit 2
        ;;
esac
