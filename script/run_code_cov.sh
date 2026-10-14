#!/usr/bin/env bash

set -exuo pipefail

################################################################################
# Setup and checks
################################################################################

# Assumptions:
# - LLVM toolchain is installed and is in the path (see https://apt.llvm.org/ for download instructions)
# - You've built DGL with coverage flags turned on using `cmake --preset rocm-cov`
# - lcov is installed and is in the path

# Output:
# - cpp_coverage_report.txt
# - lcov_output/ (directory with html coverage report)
# - 

# Notes:
# - This coverage report omits third party libraries
# - This coverage report is only generated from the C++ tests. The python tests (which 
#   also call functions from libdgl.so) are not included.


SCRIPT_DIR=$(realpath $(dirname $0))
DGL_DIR=$(realpath ${SCRIPT_DIR}/..)

# Add warning if llvm-cov or llvm-profdata are not in the path
if ! command -v llvm-cov &> /dev/null; then
  echo "llvm-cov could not be found in the path"
  exit 1
fi
if ! command -v llvm-profdata &> /dev/null; then
  echo "llvm-profdata could not be found in the path"
  exit 1
fi
if ! command -v lcov &> /dev/null; then
  echo "lcov could not be found in the path"
  exit 1
fi

################################################################################
# Run the python tests
################################################################################

bash ${DGL_DIR}/tests/scripts/task_unit_test_rocm.sh pytorch gpu on
# generates lcov_pytest.info trace

################################################################################
# Run tests
################################################################################

${DGL_DIR}/build/runUnitTests

################################################################################
# Generate C++ coverage report
################################################################################

# TODO do we want to collect graphbolt coverage here if we are skipping other third party libraries?
BINARIES="${DGL_DIR}/build/libdgl.so"

# Collect all the profraw files
find . -type f -name "*.profraw" > rawprofiles.list

# Merge the profraw files
llvm-profdata merge --sparse  --input-files=rawprofiles.list -o coverage.profdata

# Generate the coverage report for full codebase, non-cuda, and cuda-only
llvm-cov report ${BINARIES} \
  -instr-profile=coverage.profdata \
  --ignore-filename-regex="third_party/" | tee cpp_coverage_report.txt

# Convert to lcov format
llvm-cov export ${BINARIES} -instr-profile=coverage.profdata -format=lcov > lcov_cpp.info

################################################################################
# Generate the full DGL coverage report
################################################################################

# Merge the lcov files
lcov --ignore-errors inconsistent --ignore-errors corrupt \
  --add-tracefile lcov_cpp.info -a lcov_pytest.info -o lcov_merged.info
# Remove third party libraries
lcov --ignore-errors inconsistent \
  --remove lcov_merged.info '*/third_party/*' -o lcov_merged.info

# Generate the coverage report
genhtml --ignore-errors inconsistent -j lcov_merged.info -o lcov_output

# Logging the import artifacts
echo "HTML coverage report generated in lcov_output/index.html"
echo "C++ coverage report generated in cpp_coverage_report.txt"
echo "C++ lcov file generated in lcov_cpp.info"
echo "Python coverage report generated in python_coverage_report.txt"
echo "Python lcov file generated in lcov_pytest.info"
echo "Full DGL lcov file generated in lcov_merged.info"
