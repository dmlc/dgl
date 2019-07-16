#include <gtest/gtest.h>
#include <dgl/array.h>

using namespace dgl;
using namespace dgl::runtime;

namespace {
template <typename T>
T* Ptr(NDArray nd) {
  return static_cast<T*>(nd->data);
}

int64_t* PI64(NDArray nd) {
  return static_cast<int64_t*>(nd->data);
}

int32_t* PI32(NDArray nd) {
  return static_cast<int32_t*>(nd->data);
}

int64_t Len(NDArray nd) {
  return nd->shape[0];
}

static constexpr DLContext CTX = DLContext{kDLCPU, 0};
}

TEST(ArrayTest, TestCreate) {
  IdArray a = aten::NewIdArray(100, CTX, 32);
  ASSERT_EQ(a->dtype.bits, 32);
  ASSERT_EQ(a->shape[0], 100);

  a = aten::NewIdArray(0);
  ASSERT_EQ(a->shape[0], 0);

  std::vector<int64_t> vec = {2, 94, 232, 30};
  a = aten::VecToIdArray(vec, 32);
  ASSERT_EQ(Len(a), vec.size());
  ASSERT_EQ(a->dtype.bits, 32);
  for (int i = 0; i < Len(a); ++i) {
    ASSERT_EQ(Ptr<int32_t>(a)[i], vec[i]);
  }

  a = aten::VecToIdArray(std::vector<int32_t>());
  ASSERT_EQ(Len(a), 0);
};

TEST(ArrayTest, TestRange) {
  IdArray a = aten::Range(10, 10, 64, CTX);
  ASSERT_EQ(Len(a), 0);
  a = aten::Range(10, 20, 32, CTX);
  ASSERT_EQ(Len(a), 10);
  ASSERT_EQ(a->dtype.bits, 32);
  for (int i = 0; i < 10; ++i)
    ASSERT_EQ(Ptr<int32_t>(a)[i], i + 10);
};

TEST(ArrayTest, TestFull) {
  IdArray a = aten::Full(-100, 0, 32, CTX);
  ASSERT_EQ(Len(a), 0);
  a = aten::Full(-100, 13, 64, CTX);
  ASSERT_EQ(Len(a), 13);
  ASSERT_EQ(a->dtype.bits, 64);
  for (int i = 0; i < 13; ++i)
    ASSERT_EQ(Ptr<int64_t>(a)[i], -100);
};

TEST(ArrayTest, TestClone) {
  IdArray a = aten::Range(0, 10, 32, CTX);
  IdArray b = aten::Clone(a);
  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(PI32(b)[i], i);
  }
  PI32(b)[0] = -1;
  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(PI32(a)[i], i);
  }
};
