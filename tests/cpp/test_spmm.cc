#if !defined(_WIN32)
#include <../src/array/cpu/spmm.h>
#include <dgl/array.h>
#include <gtest/gtest.h>
#include <time.h>

#include <random>

#include "./common.h"

using namespace dgl;
using namespace dgl::runtime;

int sizes[] = {1, 7, 8, 9, 31, 32, 33, 54, 63, 64, 65, 256, 257};
namespace ns_op = dgl::aten::cpu::op;
namespace {

template <class T>
void GenerateData(T* data, int dim, T mul) {
  for (int i = 0; i < dim; i++) {
    data[i] = (i + 1) * mul;
  }
}

template <class T>
void GenerateRandomData(T* data, int dim) {
  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<> dist(0, 10000);
  for (int i = 0; i < dim; i++) {
    data[i] = (dist(rng) / 100);
  }
}

template <class T>
void GenerateZeroData(T* data, int dim) {
  for (int i = 0; i < dim; i++) {
    data[i] = 0;
  }
}

template <class T>
void Copy(T* exp, T* out, T* hs, int dim) {
  for (int i = 0; i < dim; i++) {
    exp[i] = out[i] + hs[i];
  }
}

template <class T>
void Add(T* exp, T* out, T* lhs, T* rhs, int dim) {
  for (int i = 0; i < dim; i++) {
    exp[i] = out[i] + lhs[i] + rhs[i];
  }
}

template <class T>
void Sub(T* exp, T* out, T* lhs, T* rhs, int dim) {
  for (int i = 0; i < dim; i++) {
    exp[i] = out[i] + lhs[i] - rhs[i];
  }
}

template <class T>
void Mul(T* exp, T* out, T* lhs, T* rhs, int dim) {
  for (int i = 0; i < dim; i++) {
    exp[i] = (out[i] + (lhs[i] * rhs[i]));
  }
}

template <class T>
void Div(T* exp, T* out, T* lhs, T* rhs, int dim) {
  for (int i = 0; i < dim; i++) {
    exp[i] = (out[i] + (lhs[i] / rhs[i]));
  }
}

template <class T>
void CheckResult(T* exp, T* out, int dim) {
  for (int i = 0; i < dim; i++) {
    ASSERT_TRUE(exp[i] == out[i]);
  }
}

}  // namespace

template <typename IDX>
void _TestSpmmCopyLhs() {
  for (size_t i = 0; i < sizeof(sizes) / sizeof(int); i++) {
    int dim = sizes[i];
    IDX out[dim], exp[dim], lhs[dim];
    GenerateZeroData(out, dim);
    GenerateRandomData(lhs, dim);

    // Calculation of expected output - 'exp'
    Copy(exp, out, lhs, dim);

    // Calculation of output using legacy path - 'out'
    for (int k = 0; k < dim; k++) {
      out[k] += ns_op::CopyLhs<IDX>::Call(lhs + k, nullptr);
    }

    CheckResult(exp, out, dim);
  }
}

TEST(SpmmTest, TestSpmmCopyLhs) {
  _TestSpmmCopyLhs<float>();
  _TestSpmmCopyLhs<double>();
  _TestSpmmCopyLhs<BFloat16>();
}

template <typename IDX>
void _TestSpmmCopyRhs() {
  for (size_t i = 0; i < sizeof(sizes) / sizeof(int); i++) {
    int dim = sizes[i];
    IDX out[dim], exp[dim], rhs[dim];
    GenerateZeroData(out, dim);
    GenerateRandomData(rhs, dim);

    // Calculation of expected output - 'exp'
    Copy(exp, out, rhs, dim);

    // Calculation of output using legacy path - 'out'
    for (int k = 0; k < dim; k++) {
      out[k] += ns_op::CopyRhs<IDX>::Call(nullptr, rhs + k);
    }

    CheckResult(exp, out, dim);
  }
}

TEST(SpmmTest, TestSpmmCopyRhs) {
  _TestSpmmCopyRhs<float>();
  _TestSpmmCopyRhs<double>();
  _TestSpmmCopyRhs<BFloat16>();
}

template <typename IDX>
void _TestSpmmAdd() {
  for (size_t i = 0; i < sizeof(sizes) / sizeof(int); i++) {
    int dim = sizes[i];
    IDX out[dim], exp[dim], lhs[dim], rhs[dim];
    GenerateZeroData(out, dim);
    GenerateRandomData(lhs, dim);
    GenerateRandomData(rhs, dim);

    // Calculation of expected output - 'exp'
    Add(exp, out, lhs, rhs, dim);

    // Calculation of output using legacy path - 'out'
    for (int k = 0; k < dim; k++) {
      out[k] += ns_op::Add<IDX>::Call(lhs + k, rhs + k);
    }

    CheckResult(exp, out, dim);
  }
}

TEST(SpmmTest, TestSpmmAdd) {
  _TestSpmmAdd<float>();
  _TestSpmmAdd<double>();
  _TestSpmmAdd<BFloat16>();
}

template <typename IDX>
void _TestSpmmSub() {
  for (size_t i = 0; i < sizeof(sizes) / sizeof(int); i++) {
    int dim = sizes[i];
    IDX out[dim], exp[dim], lhs[dim], rhs[dim];
    GenerateZeroData(out, dim);
    GenerateRandomData(lhs, dim);
    GenerateRandomData(rhs, dim);

    // Calculation of expected output - 'exp'
    Sub(exp, out, lhs, rhs, dim);

    // Calculation of output using legacy path - 'out'
    for (int k = 0; k < dim; k++) {
      out[k] += ns_op::Sub<IDX>::Call(lhs + k, rhs + k);
    }

    CheckResult(exp, out, dim);
  }
}

TEST(SpmmTest, TestSpmmSub) {
  _TestSpmmSub<float>();
  _TestSpmmSub<double>();
  _TestSpmmSub<BFloat16>();
}

template <typename IDX>
void _TestSpmmMul() {
  for (size_t i = 0; i < sizeof(sizes) / sizeof(int); i++) {
    int dim = sizes[i];
    IDX out[dim], exp[dim], lhs[dim], rhs[dim];
    GenerateZeroData(out, dim);
    GenerateRandomData(lhs, dim);
    GenerateRandomData(rhs, dim);

    // Calculation of expected output - 'exp'
    Mul(exp, out, lhs, rhs, dim);

    // Calculation of output using legacy path - 'out'
    for (int k = 0; k < dim; k++) {
      out[k] += ns_op::Mul<IDX>::Call(lhs + k, rhs + k);
    }

    CheckResult(exp, out, dim);
  }
}

TEST(SpmmTest, TestSpmmMul) {
  _TestSpmmMul<float>();
  _TestSpmmMul<double>();
  _TestSpmmMul<BFloat16>();
}

template <typename IDX>
void _TestSpmmDiv() {
  for (size_t i = 0; i < sizeof(sizes) / sizeof(int); i++) {
    int dim = sizes[i];
    IDX out[dim], exp[dim], lhs[dim], rhs[dim];
    GenerateZeroData(out, dim);
    GenerateData(lhs, dim, (IDX)15);
    GenerateData(rhs, dim, (IDX)1);

    // Calculation of expected output - 'exp'
    Div(exp, out, lhs, rhs, dim);

    // Calculation of output using legacy path - 'out'
    for (int k = 0; k < dim; k++) {
      out[k] += ns_op::Div<IDX>::Call(lhs + k, rhs + k);
    }

    CheckResult(exp, out, dim);
  }
}

TEST(SpmmTest, TestSpmmDiv) {
  _TestSpmmDiv<float>();
  _TestSpmmDiv<double>();
  _TestSpmmDiv<BFloat16>();
}
#endif  // _WIN32
