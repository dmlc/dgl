#if !defined(_WIN32)
#ifdef USE_AVX
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
void CheckResult(T* exp, T* out, T* out_intel_kernel, int dim) {
  for (int i = 0; i < dim; i++) {
    ASSERT_TRUE(exp[i] == out[i]);
    if (out_intel_kernel != nullptr) {
      ASSERT_TRUE(out[i] == out_intel_kernel[i]);
    }
  }
}

}  // namespace

template <class ElemWiseUpd>
ElemWiseUpd* generic_ElemWiseUpd() {
  static std::unique_ptr<ElemWiseUpd> asm_kernel_ptr(
      (dgl::IntelKernel<>::IsEnabled()) ? new ElemWiseUpd() : nullptr);
  ElemWiseUpd* cpu_spec = (asm_kernel_ptr && asm_kernel_ptr->applicable())
                              ? asm_kernel_ptr.get()
                              : nullptr;

  return cpu_spec;
}

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

    // Calculation of output using intel path - 'out_intel_kernel'
    auto* cpu_spec =
        generic_ElemWiseUpd<dgl::ElemWiseAddUpdate<ns_op::CopyLhs<IDX>>>();
    if (cpu_spec) {
      IDX out_intel_kernel[dim];
      GenerateZeroData(out_intel_kernel, dim);
      cpu_spec->run(out_intel_kernel, lhs, nullptr, dim);
      CheckResult(exp, out, out_intel_kernel, dim);
    } else {
      IDX* out_intel_kernel = nullptr;
      CheckResult(exp, out, out_intel_kernel, dim);
    }
  }
}

TEST(SpmmTest, TestSpmmCopyLhs) {
  _TestSpmmCopyLhs<float>();
  _TestSpmmCopyLhs<double>();
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

    // Calculation of output using intel path - 'out_intel_kernel'
    auto* cpu_spec =
        generic_ElemWiseUpd<dgl::ElemWiseAddUpdate<ns_op::CopyRhs<IDX>>>();
    if (cpu_spec) {
      IDX out_intel_kernel[dim];
      GenerateZeroData(out_intel_kernel, dim);
      cpu_spec->run(out_intel_kernel, nullptr, rhs, dim);
      CheckResult(exp, out, out_intel_kernel, dim);
    } else {
      IDX* out_intel_kernel = nullptr;
      CheckResult(exp, out, out_intel_kernel, dim);
    }
  }
}

TEST(SpmmTest, TestSpmmCopyRhs) {
  _TestSpmmCopyRhs<float>();
  _TestSpmmCopyRhs<double>();
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

    // Calculation of output using intel path - 'out_intel_kernel'
    auto* cpu_spec =
        generic_ElemWiseUpd<dgl::ElemWiseAddUpdate<ns_op::Add<IDX>>>();
    if (cpu_spec) {
      IDX out_intel_kernel[dim];
      GenerateZeroData(out_intel_kernel, dim);
      cpu_spec->run(out_intel_kernel, lhs, rhs, dim);
      CheckResult(exp, out, out_intel_kernel, dim);
    } else {
      IDX* out_intel_kernel = nullptr;
      CheckResult(exp, out, out_intel_kernel, dim);
    }
  }
}

TEST(SpmmTest, TestSpmmAdd) {
  _TestSpmmAdd<float>();
  _TestSpmmAdd<double>();
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

    // Calculation of output using intel path - 'out_intel_kernel'
    auto* cpu_spec =
        generic_ElemWiseUpd<dgl::ElemWiseAddUpdate<ns_op::Sub<IDX>>>();
    if (cpu_spec) {
      IDX out_intel_kernel[dim];
      GenerateZeroData(out_intel_kernel, dim);
      cpu_spec->run(out_intel_kernel, lhs, rhs, dim);
      CheckResult(exp, out, out_intel_kernel, dim);
    } else {
      IDX* out_intel_kernel = nullptr;
      CheckResult(exp, out, out_intel_kernel, dim);
    }
  }
}

TEST(SpmmTest, TestSpmmSub) {
  _TestSpmmSub<float>();
  _TestSpmmSub<double>();
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

    // Calculation of output using intel path - 'out_intel_kernel'
    auto* cpu_spec =
        generic_ElemWiseUpd<dgl::ElemWiseAddUpdate<ns_op::Mul<IDX>>>();
    if (cpu_spec) {
      IDX out_intel_kernel[dim];
      GenerateZeroData(out_intel_kernel, dim);
      cpu_spec->run(out_intel_kernel, lhs, rhs, dim);
      CheckResult(exp, out, out_intel_kernel, dim);
    } else {
      IDX* out_intel_kernel = nullptr;
      CheckResult(exp, out, out_intel_kernel, dim);
    }
  }
}

TEST(SpmmTest, TestSpmmMul) {
  _TestSpmmMul<float>();
  _TestSpmmMul<double>();
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

    // Calculation of output using intel path - 'out_intel_kernel'
    auto* cpu_spec =
        generic_ElemWiseUpd<dgl::ElemWiseAddUpdate<ns_op::Div<IDX>>>();
    if (cpu_spec) {
      IDX out_intel_kernel[dim];
      GenerateZeroData(out_intel_kernel, dim);
      cpu_spec->run(out_intel_kernel, lhs, rhs, dim);
      CheckResult(exp, out, out_intel_kernel, dim);
    } else {
      IDX* out_intel_kernel = nullptr;
      CheckResult(exp, out, out_intel_kernel, dim);
    }
  }
}

TEST(SpmmTest, TestSpmmDiv) {
  _TestSpmmDiv<float>();
  _TestSpmmDiv<double>();
}
#endif  // USE_AVX
#endif  // _WIN32
