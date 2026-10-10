#ifdef DGL_USE_ASCEND
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <torch/extension.h>

#include "ascend/concat_tiling.h"
#endif
/**
 *  Copyright (c) 2019-2022 by Contributors
 * @file array/array.cc
 * @brief DGL array utilities implementation
 */
#include <dgl/array.h>
#include <dgl/bcast.h>
#include <dgl/graph_traversal.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/device_api.h>
#include <dgl/runtime/shared_mem.h>

#include <sstream>

#include "../c_api_common.h"
#include "./arith.h"
#include "./array_op.h"
#include "./kernel_decl.h"

using namespace dgl::runtime;

namespace dgl {
namespace aten {

IdArray NewIdArray(int64_t length, DGLContext ctx, uint8_t nbits) {
  return IdArray::Empty({length}, DGLDataType{kDGLInt, nbits, 1}, ctx);
}

FloatArray NewFloatArray(int64_t length, DGLContext ctx, uint8_t nbits) {
  return FloatArray::Empty({length}, DGLDataType{kDGLFloat, nbits, 1}, ctx);
}

IdArray Clone(IdArray arr) {
  IdArray ret = NewIdArray(arr->shape[0], arr->ctx, arr->dtype.bits);
  ret.CopyFrom(arr);
  return ret;
}

IdArray Range(int64_t low, int64_t high, uint8_t nbits, DGLContext ctx) {
  IdArray ret;
#ifdef DGL_USE_ASCEND
  if (ctx.device_type == kDGLAscend) {
    if (nbits == 32) {
      ret = impl::Range<kDGLAscend, int32_t>(
          static_cast<int32_t>(low), static_cast<int32_t>(high), ctx);
    } else if (nbits == 64) {
      ret = impl::Range<kDGLAscend, int64_t>(
          static_cast<int64_t>(low), static_cast<int64_t>(high), ctx);
    } else {
      LOG(FATAL) << "Only int32 or int64 is supported.";
    }
    return ret;
  }
#endif
  ATEN_XPU_SWITCH_CUDA(ctx.device_type, XPU, "Range", {
    if (nbits == 32) {
      ret = impl::Range<XPU, int32_t>(low, high, ctx);
    } else if (nbits == 64) {
      ret = impl::Range<XPU, int64_t>(low, high, ctx);
    } else {
      LOG(FATAL) << "Only int32 or int64 is supported.";
    }
  });
  return ret;
}

IdArray Full(int64_t val, int64_t length, uint8_t nbits, DGLContext ctx) {
  IdArray ret;

#ifdef DGL_USE_ASCEND
  if (ctx.device_type == kDGLAscend) {
    if (nbits == 32) {
      ret = impl::Full<kDGLAscend, int32_t>(static_cast<int32_t>(val), length, ctx);
    } else if (nbits == 64) {
      ret = impl::Full<kDGLAscend, int64_t>(val, length, ctx);
    } else {
      LOG(FATAL) << "Only int32 or int64 is supported.";
    }
    return ret;
  }
#endif

  ATEN_XPU_SWITCH_CUDA(ctx.device_type, XPU, "Full", {
    if (nbits == 32) {
      ret = impl::Full<XPU, int32_t>(val, length, ctx);
    } else if (nbits == 64) {
      ret = impl::Full<XPU, int64_t>(val, length, ctx);
    } else {
      LOG(FATAL) << "Only int32 or int64 is supported.";
    }
  });
  return ret;
}

template <typename DType>
NDArray Full(DType val, int64_t length, DGLContext ctx) {
  NDArray ret;

#ifdef DGL_USE_ASCEND
  if (ctx.device_type == kDGLAscend) {
    return impl::Full<kDGLAscend, DType>(val, length, ctx);
  }
#endif

  ATEN_XPU_SWITCH_CUDA(ctx.device_type, XPU, "Full", {
    ret = impl::Full<XPU, DType>(val, length, ctx);
  });
  return ret;
}

template NDArray Full<int32_t>(int32_t val, int64_t length, DGLContext ctx);
template NDArray Full<int64_t>(int64_t val, int64_t length, DGLContext ctx);
template NDArray Full<float>(float val, int64_t length, DGLContext ctx);
template NDArray Full<double>(double val, int64_t length, DGLContext ctx);

IdArray AsNumBits(IdArray arr, uint8_t bits) {
  CHECK(bits == 32 || bits == 64)
      << "Invalid ID type. Must be int32 or int64, but got int"
      << static_cast<int>(bits) << ".";
  if (arr->dtype.bits == bits) return arr;
  if (arr.NumElements() == 0) return NewIdArray(arr->shape[0], arr->ctx, bits);
  IdArray ret;
#ifdef DGL_USE_ASCEND
  if (arr->ctx.device_type == kDGLAscend) {
    ATEN_ID_TYPE_SWITCH(
        arr->dtype, IdType, { ret = impl::AsNumBits<kDGLAscend, IdType>(arr, bits); });
    return ret;
  }
#endif
  ATEN_XPU_SWITCH_CUDA(arr->ctx.device_type, XPU, "AsNumBits", {
    ATEN_ID_TYPE_SWITCH(
        arr->dtype, IdType, { ret = impl::AsNumBits<XPU, IdType>(arr, bits); });
  });
  return ret;
}

IdArray HStack(IdArray lhs, IdArray rhs) {
  IdArray ret;
  CHECK_SAME_CONTEXT(lhs, rhs);
  CHECK_SAME_DTYPE(lhs, rhs);
  CHECK_EQ(lhs->shape[0], rhs->shape[0]);
  auto device = runtime::DeviceAPI::Get(lhs->ctx);
  const auto& ctx = lhs->ctx;
  ATEN_ID_TYPE_SWITCH(lhs->dtype, IdType, {
    const int64_t len = lhs->shape[0];
    ret = NewIdArray(2 * len, lhs->ctx, lhs->dtype.bits);
    device->CopyDataFromTo(
        lhs.Ptr<IdType>(), 0, ret.Ptr<IdType>(), 0, len * sizeof(IdType), ctx,
        ctx, lhs->dtype);
    device->CopyDataFromTo(
        rhs.Ptr<IdType>(), 0, ret.Ptr<IdType>(), len * sizeof(IdType),
        len * sizeof(IdType), ctx, ctx, lhs->dtype);
  });
  return ret;
}

NDArray IndexSelect(NDArray array, IdArray index) {
  NDArray ret;
  CHECK_GE(array->ndim, 1) << "Only support array with at least 1 dimension";
  CHECK_EQ(index->ndim, 1) << "Index array must be an 1D array.";
  // if array is not pinned, index has the same context as array
  // if array is pinned, op dispatching depends on the context of index
  CHECK_VALID_CONTEXT(array, index);
#ifdef DGL_USE_ASCEND
  if (index->ctx.device_type == kDGLAscend ||
      array->ctx.device_type == kDGLAscend) {
    ATEN_DTYPE_SWITCH(array->dtype, DType, "values", {
      ATEN_ID_TYPE_SWITCH(index->dtype, IdType, {
        ret = impl::IndexSelect<kDGLAscend, DType, IdType>(
            array, index);
      });
    });
  } else
#endif
  {
    ATEN_XPU_SWITCH_CUDA(index->ctx.device_type, XPU, "IndexSelect", {
      ATEN_DTYPE_SWITCH(array->dtype, DType, "values", {
        ATEN_ID_TYPE_SWITCH(index->dtype, IdType, {
          ret = impl::IndexSelect<XPU, DType, IdType>(array, index);
        });
      });
    });
  }
  return ret;
}

template <typename ValueType>
ValueType IndexSelect(NDArray array, int64_t index) {
  CHECK_EQ(array->ndim, 1) << "Only support select values from 1D array.";
  CHECK(index >= 0 && index < array.NumElements())
      << "Index " << index << " is out of bound.";
  ValueType ret = 0;
#ifdef DGL_USE_ASCEND
  if (array->ctx.device_type == kDGLAscend) {
    ATEN_DTYPE_SWITCH(array->dtype, DType, "values", {
      ret = impl::IndexSelect<kDGLAscend, DType>(array, index);
    });
  } else
#endif
  {
    ATEN_XPU_SWITCH_CUDA(array->ctx.device_type, XPU, "IndexSelect", {
      ATEN_DTYPE_SWITCH(array->dtype, DType, "values", {
        ret = impl::IndexSelect<XPU, DType>(array, index);
      });
    });
  }
  return ret;
}
template int32_t IndexSelect<int32_t>(NDArray array, int64_t index);
template int64_t IndexSelect<int64_t>(NDArray array, int64_t index);
template uint32_t IndexSelect<uint32_t>(NDArray array, int64_t index);
template uint64_t IndexSelect<uint64_t>(NDArray array, int64_t index);
template float IndexSelect<float>(NDArray array, int64_t index);
template double IndexSelect<double>(NDArray array, int64_t index);

NDArray IndexSelect(NDArray array, int64_t start, int64_t end) {
  CHECK_EQ(array->ndim, 1) << "Only support select values from 1D array.";
  CHECK(start >= 0 && start < array.NumElements())
      << "Index " << start << " is out of bound.";
  CHECK(end >= 0 && end <= array.NumElements())
      << "Index " << end << " is out of bound.";
  CHECK_LE(start, end);
  auto device = runtime::DeviceAPI::Get(array->ctx);
  const int64_t len = end - start;
  NDArray ret = NDArray::Empty({len}, array->dtype, array->ctx);
  ATEN_DTYPE_SWITCH(array->dtype, DType, "values", {
    device->CopyDataFromTo(
        array->data, start * sizeof(DType), ret->data, 0, len * sizeof(DType),
        array->ctx, ret->ctx, array->dtype);
  });
  return ret;
}

NDArray Scatter(NDArray array, IdArray indices) {
  NDArray ret;
  ATEN_XPU_SWITCH(array->ctx.device_type, XPU, "Scatter", {
    ATEN_DTYPE_SWITCH(array->dtype, DType, "values", {
      ATEN_ID_TYPE_SWITCH(indices->dtype, IdType, {
        ret = impl::Scatter<XPU, DType, IdType>(array, indices);
      });
    });
  });
  return ret;
}

void Scatter_(IdArray index, NDArray value, NDArray out) {
  CHECK_SAME_DTYPE(value, out);
  CHECK_SAME_CONTEXT(index, value);
  CHECK_SAME_CONTEXT(index, out);
  CHECK_EQ(value->shape[0], index->shape[0]);
  if (index->shape[0] == 0) return;
  if (value->ctx.device_type == kDGLAscend) {
    DGLContext cpu_ctx{kDGLCPU, 0};
    auto index_cpu = index.CopyTo(cpu_ctx);
    auto value_cpu = value.CopyTo(cpu_ctx);
    auto out_cpu = out.CopyTo(cpu_ctx);
    ATEN_DTYPE_SWITCH(value->dtype, DType, "values", {
      ATEN_ID_TYPE_SWITCH(index->dtype, IdType, {
        impl::Scatter_<kDGLCPU, DType, IdType>(index_cpu, value_cpu, out_cpu);
      });
    });
    out.CopyFrom(out_cpu);
  } else {
    ATEN_XPU_SWITCH_CUDA(value->ctx.device_type, XPU, "Scatter_", {
      ATEN_DTYPE_SWITCH(value->dtype, DType, "values", {
        ATEN_ID_TYPE_SWITCH(index->dtype, IdType, {
          impl::Scatter_<XPU, DType, IdType>(index, value, out);
        });
      });
    });
  }
}

NDArray Repeat(NDArray array, IdArray repeats) {
  NDArray ret;
  ATEN_XPU_SWITCH(array->ctx.device_type, XPU, "Repeat", {
    ATEN_DTYPE_SWITCH(array->dtype, DType, "values", {
      ATEN_ID_TYPE_SWITCH(repeats->dtype, IdType, {
        ret = impl::Repeat<XPU, DType, IdType>(array, repeats);
      });
    });
  });
  return ret;
}

IdArray Relabel_(const std::vector<IdArray>& arrays) {
  IdArray ret;
  if (arrays[0]->ctx.device_type == kDGLAscend) {
    DGLContext cpu_ctx{kDGLCPU, 0};
    std::vector<IdArray> arrays_cpu;
    arrays_cpu.reserve(arrays.size());
    for (const auto& arr : arrays) {
      arrays_cpu.push_back(arr.CopyTo(cpu_ctx));
    }
    ATEN_ID_TYPE_SWITCH(arrays[0]->dtype, IdType, {
      ret = impl::Relabel_<kDGLCPU, IdType>(arrays_cpu);
    });
    ret = ret.CopyTo(arrays[0]->ctx);
  } else {
    ATEN_XPU_SWITCH_CUDA(arrays[0]->ctx.device_type, XPU, "Relabel_", {
      ATEN_ID_TYPE_SWITCH(arrays[0]->dtype, IdType, {
        ret = impl::Relabel_<XPU, IdType>(arrays);
      });
    });
  }
  return ret;
}

NDArray Concat(const std::vector<IdArray>& arrays) {
  int64_t len = 0;
  for (size_t i = 0; i < arrays.size(); ++i) {
    len += arrays[i]->shape[0];
    CHECK_SAME_DTYPE(arrays[0], arrays[i]);
    CHECK_SAME_CONTEXT(arrays[0], arrays[i]);
  }

#ifdef DGL_USE_ASCEND
  // 1D IdArray 拼接等价于 ConcatND(arrays, dim=0)。对齐检查与 dtype 分发由
  // ConcatND 内部完成，非对齐输入或超过 kernel 2^32 元素上限时自动回退通用路径，
  // 保证与旧通用路径（逐段 CopyDataFromTo）行为一致。
  if (arrays[0]->ctx.device_type == kDGLAscend &&
      arrays[0]->dtype.code == kDGLInt &&
      (arrays[0]->dtype.bits == 32 || arrays[0]->dtype.bits == 64) &&
      len < (static_cast<int64_t>(1) << 32)) {
    std::vector<NDArray> nds(arrays.begin(), arrays.end());
    return ConcatND(nds, 0);
  }
#endif

  NDArray ret_arr = NDArray::Empty({len}, arrays[0]->dtype, arrays[0]->ctx);

  auto device = runtime::DeviceAPI::Get(arrays[0]->ctx);
  int64_t offset = 0;
  for (size_t i = 0; i < arrays.size(); ++i) {
    ATEN_DTYPE_SWITCH(arrays[i]->dtype, DType, "array", {
      device->CopyDataFromTo(
          static_cast<DType*>(arrays[i]->data), 0,
          static_cast<DType*>(ret_arr->data), offset,
          arrays[i]->shape[0] * sizeof(DType), arrays[i]->ctx, ret_arr->ctx,
          arrays[i]->dtype);

      offset += arrays[i]->shape[0] * sizeof(DType);
    });
  }

  return ret_arr;
}

#ifdef DGL_USE_ASCEND
namespace {

// The AscendC kernel moves data with DataCopyPad, which requires 32-byte-aligned
// GM addresses. Freshly allocated buffers are 512-byte aligned on Ascend, but a
// DLPack tensor may be a slice of a larger allocation and hence start anywhere,
// so misaligned inputs must fall back to the byte-wise copy path.
bool AscendDataCopyAligned(
    const std::vector<NDArray>& arrays, const NDArray& out) {
  auto aligned = [](const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & (CONCAT_ALIGN_BYTES - 1)) == 0;
  };
  if (!aligned(out->data)) return false;
  for (const NDArray& a : arrays) {
    if (!aligned(a->data)) return false;
  }
  return true;
}

}  // namespace
#endif  // DGL_USE_ASCEND

NDArray ConcatND(const std::vector<NDArray>& arrays, int64_t dim) {
  CHECK(!arrays.empty()) << "ConcatND requires at least one array.";
  const NDArray& first = arrays[0];
  const int64_t ndim = first->ndim;
  CHECK_GE(ndim, 1) << "ConcatND requires ndim >= 1, got " << ndim << ".";
  const int64_t axis = dim < 0 ? dim + ndim : dim;
  CHECK_GE(axis, 0) << "ConcatND: dim " << dim << " is out of range for ndim "
                    << ndim << ".";
  CHECK_LT(axis, ndim) << "ConcatND: dim " << dim << " is out of range for ndim "
                       << ndim << ".";

  std::vector<int64_t> out_shape(first->shape, first->shape + ndim);
  out_shape[axis] = 0;
  for (size_t i = 0; i < arrays.size(); ++i) {
    CHECK_SAME_DTYPE(first, arrays[i]);
    CHECK_SAME_CONTEXT(first, arrays[i]);
    CHECK_NDIM(arrays[i], ndim, "arrays[i]");
    for (int64_t d = 0; d < ndim; ++d) {
      if (d == axis) {
        out_shape[d] += arrays[i]->shape[d];
      } else {
        CHECK_EQ(first->shape[d], arrays[i]->shape[d])
            << "ConcatND: all shapes must match except on dim " << axis
            << ", but arrays[0] and arrays[" << i << "] differ on axis " << d
            << " (" << first->shape[d] << " vs " << arrays[i]->shape[d] << ").";
      }
    }
  }

  NDArray ret = NDArray::Empty(out_shape, first->dtype, first->ctx);
  if (out_shape[axis] == 0) return ret;

#ifdef DGL_USE_ASCEND
  if (first->ctx.device_type == kDGLAscend &&
      AscendDataCopyAligned(arrays, ret)) {
    if (IS_FLOAT32(first)) {
      impl::ConcatND<kDGLAscend, float>(arrays, ret, axis);
      return ret;
    }
    if (first->dtype.code == kDGLFloat && first->dtype.bits == 16) {
      // fp16 travels as uint16_t on Ascend; the kernel only moves bytes.
      impl::ConcatND<kDGLAscend, uint16_t>(arrays, ret, axis);
      return ret;
    }
    // int32/int64: 同一搬运 kernel 覆盖 1D IdArray 拼接（aten::Concat）与整型特征。
    if (first->dtype.code == kDGLInt && first->dtype.bits == 32) {
      impl::ConcatND<kDGLAscend, int32_t>(arrays, ret, axis);
      return ret;
    }
    if (first->dtype.code == kDGLInt && first->dtype.bits == 64) {
      impl::ConcatND<kDGLAscend, int64_t>(arrays, ret, axis);
      return ret;
    }
  }
#endif

  // Generic path: one device copy per (input, outer segment). Correct on every
  // device and dtype; dtypes without an Ascend kernel take this route too.
  const int64_t elem_bytes = (first->dtype.bits * first->dtype.lanes + 7) / 8;
  int64_t outer = 1, inner = 1;
  for (int64_t d = 0; d < axis; ++d) outer *= first->shape[d];
  for (int64_t d = axis + 1; d < ndim; ++d) inner *= first->shape[d];
  const int64_t dst_stride = out_shape[axis] * inner * elem_bytes;

  auto device = runtime::DeviceAPI::Get(first->ctx);
  int64_t dst_base = 0;
  for (size_t i = 0; i < arrays.size(); ++i) {
    const int64_t seg_bytes = arrays[i]->shape[axis] * inner * elem_bytes;
    if (seg_bytes > 0) {
      for (int64_t s = 0; s < outer; ++s) {
        device->CopyDataFromTo(
            arrays[i]->data, s * seg_bytes, ret->data, dst_base + s * dst_stride,
            seg_bytes, arrays[i]->ctx, ret->ctx, arrays[i]->dtype);
      }
    }
    dst_base += seg_bytes;
  }

  return ret;
}

template <typename ValueType>
std::tuple<NDArray, IdArray, IdArray> Pack(NDArray array, ValueType pad_value) {
  std::tuple<NDArray, IdArray, IdArray> ret;
  ATEN_XPU_SWITCH(array->ctx.device_type, XPU, "Pack", {
    ATEN_DTYPE_SWITCH(array->dtype, DType, "array", {
      ret = impl::Pack<XPU, DType>(array, static_cast<DType>(pad_value));
    });
  });
  return ret;
}

template std::tuple<NDArray, IdArray, IdArray> Pack<int32_t>(NDArray, int32_t);
template std::tuple<NDArray, IdArray, IdArray> Pack<int64_t>(NDArray, int64_t);
template std::tuple<NDArray, IdArray, IdArray> Pack<uint32_t>(
    NDArray, uint32_t);
template std::tuple<NDArray, IdArray, IdArray> Pack<uint64_t>(
    NDArray, uint64_t);
template std::tuple<NDArray, IdArray, IdArray> Pack<float>(NDArray, float);
template std::tuple<NDArray, IdArray, IdArray> Pack<double>(NDArray, double);

std::pair<NDArray, IdArray> ConcatSlices(NDArray array, IdArray lengths) {
  std::pair<NDArray, IdArray> ret;
  ATEN_XPU_SWITCH(array->ctx.device_type, XPU, "ConcatSlices", {
    ATEN_DTYPE_SWITCH(array->dtype, DType, "array", {
      ATEN_ID_TYPE_SWITCH(lengths->dtype, IdType, {
        ret = impl::ConcatSlices<XPU, DType, IdType>(array, lengths);
      });
    });
  });
  return ret;
}

IdArray CumSum(IdArray array, bool prepend_zero) {
  IdArray ret;
  ATEN_XPU_SWITCH_CUDA(array->ctx.device_type, XPU, "CumSum", {
    ATEN_ID_TYPE_SWITCH(array->dtype, IdType, {
      ret = impl::CumSum<XPU, IdType>(array, prepend_zero);
    });
  });
  return ret;
}

IdArray NonZero(NDArray array) {
  IdArray ret;
  ATEN_XPU_SWITCH_CUDA(array->ctx.device_type, XPU, "NonZero", {
    ATEN_ID_TYPE_SWITCH(
        array->dtype, DType, { ret = impl::NonZero<XPU, DType>(array); });
  });
  return ret;
}

std::pair<IdArray, IdArray> Sort(IdArray array, const int num_bits) {
  if (array.NumElements() == 0) {
    IdArray idx = NewIdArray(0, array->ctx, 64);
    return std::make_pair(array, idx);
  }
  std::pair<IdArray, IdArray> ret;
  if (array->ctx.device_type == kDGLAscend) {
    DGLContext cpu_ctx{kDGLCPU, 0};
    auto array_cpu = array.CopyTo(cpu_ctx);
    ATEN_ID_TYPE_SWITCH(array->dtype, IdType, {
      ret = impl::Sort<kDGLCPU, IdType>(array_cpu, num_bits);
    });
    ret.first = ret.first.CopyTo(array->ctx);
    ret.second = ret.second.CopyTo(array->ctx);
  } else {
    ATEN_XPU_SWITCH_CUDA(array->ctx.device_type, XPU, "Sort", {
      ATEN_ID_TYPE_SWITCH(array->dtype, IdType, {
        ret = impl::Sort<XPU, IdType>(array, num_bits);
      });
    });
  }
  return ret;
}

std::string ToDebugString(NDArray array) {
  std::ostringstream oss;
  NDArray a = array.CopyTo(DGLContext{kDGLCPU, 0});
  oss << "array([";
  ATEN_DTYPE_SWITCH(a->dtype, DType, "array", {
    for (int64_t i = 0; i < std::min<int64_t>(a.NumElements(), 10L); ++i) {
      oss << a.Ptr<DType>()[i] << ", ";
    }
  });
  if (a.NumElements() > 10) oss << "...";
  oss << "], dtype=" << array->dtype << ", ctx=" << array->ctx << ")";
  return oss.str();
}

///////////////////////// CSR routines //////////////////////////

bool CSRIsNonZero(CSRMatrix csr, int64_t row, int64_t col) {
  CHECK(row >= 0 && row < csr.num_rows) << "Invalid row index: " << row;
  CHECK(col >= 0 && col < csr.num_cols) << "Invalid col index: " << col;
  bool ret = false;
  ATEN_CSR_SWITCH_CUDA(csr, XPU, IdType, "CSRIsNonZero", {
    ret = impl::CSRIsNonZero<XPU, IdType>(csr, row, col);
  });
  return ret;
}

NDArray CSRIsNonZero(CSRMatrix csr, NDArray row, NDArray col) {
  NDArray ret;
  CHECK_SAME_DTYPE(csr.indices, row);
  CHECK_SAME_DTYPE(csr.indices, col);
  CHECK_SAME_CONTEXT(row, col);
  ATEN_CSR_SWITCH_CUDA_UVA(csr, row, XPU, IdType, "CSRIsNonZero", {
    ret = impl::CSRIsNonZero<XPU, IdType>(csr, row, col);
  });
  return ret;
}

bool CSRHasDuplicate(CSRMatrix csr) {
  bool ret = false;
  ATEN_CSR_SWITCH_CUDA(csr, XPU, IdType, "CSRHasDuplicate", {
    ret = impl::CSRHasDuplicate<XPU, IdType>(csr);
  });
  return ret;
}

int64_t CSRGetRowNNZ(CSRMatrix csr, int64_t row) {
  CHECK(row >= 0 && row < csr.num_rows) << "Invalid row index: " << row;
  int64_t ret = 0;
#ifdef DGL_USE_ASCEND
  if (csr.indptr->ctx.device_type == kDGLAscend) {
    ATEN_ID_TYPE_SWITCH(csr.indptr->dtype, IdType, {
      ret = impl::CSRGetRowNNZ<kDGLAscend, IdType>(csr, row);
    });
  } else
#endif
  {
    ATEN_CSR_SWITCH_CUDA(csr, XPU, IdType, "CSRGetRowNNZ", {
      ret = impl::CSRGetRowNNZ<XPU, IdType>(csr, row);
    });
  }
  return ret;
}

NDArray CSRGetRowNNZ(CSRMatrix csr, NDArray row) {
  NDArray ret;
  CHECK_SAME_DTYPE(csr.indices, row);
#ifdef DGL_USE_ASCEND
  if (csr.indptr->ctx.device_type == kDGLAscend) {
    ATEN_ID_TYPE_SWITCH(csr.indptr->dtype, IdType, {
      ret = impl::CSRGetRowNNZ<kDGLAscend, IdType>(csr, row);
    });
  } else
#endif
  {
    ATEN_CSR_SWITCH_CUDA_UVA(csr, row, XPU, IdType, "CSRGetRowNNZ", {
      ret = impl::CSRGetRowNNZ<XPU, IdType>(csr, row);
    });
  }
  return ret;
}

NDArray CSRGetRowColumnIndices(CSRMatrix csr, int64_t row) {
  CHECK(row >= 0 && row < csr.num_rows) << "Invalid row index: " << row;
  NDArray ret;
#ifdef DGL_USE_ASCEND
  if (csr.indptr->ctx.device_type == kDGLAscend) {
    ATEN_ID_TYPE_SWITCH(csr.indptr->dtype, IdType, {
      ret = impl::CSRGetRowColumnIndices<kDGLAscend, IdType>(csr, row);
    });
  } else
#endif
  {
    ATEN_CSR_SWITCH_CUDA(csr, XPU, IdType, "CSRGetRowColumnIndices", {
      ret = impl::CSRGetRowColumnIndices<XPU, IdType>(csr, row);
    });
  }
  return ret;
}

NDArray CSRGetRowData(CSRMatrix csr, int64_t row) {
  CHECK(row >= 0 && row < csr.num_rows) << "Invalid row index: " << row;
  NDArray ret;
#ifdef DGL_USE_ASCEND
  if (csr.indptr->ctx.device_type == kDGLAscend) {
    ATEN_ID_TYPE_SWITCH(csr.indptr->dtype, IdType, {
      ret = impl::CSRGetRowData<kDGLAscend, IdType>(csr, row);
    });
  } else
#endif
  {
    ATEN_CSR_SWITCH_CUDA(csr, XPU, IdType, "CSRGetRowData", {
      ret = impl::CSRGetRowData<XPU, IdType>(csr, row);
    });
  }
  return ret;
}

bool CSRIsSorted(CSRMatrix csr) {
  if (csr.indices->shape[0] <= 1) return true;
  bool ret = false;
  ATEN_CSR_SWITCH_CUDA(csr, XPU, IdType, "CSRIsSorted", {
    ret = impl::CSRIsSorted<XPU, IdType>(csr);
  });
  return ret;
}

NDArray CSRGetData(CSRMatrix csr, NDArray rows, NDArray cols) {
  NDArray ret;
  CHECK_SAME_DTYPE(csr.indices, rows);
  CHECK_SAME_DTYPE(csr.indices, cols);
  CHECK_SAME_CONTEXT(rows, cols);
#ifdef DGL_USE_ASCEND
  if (rows->ctx.device_type == kDGLAscend) {
    ATEN_ID_TYPE_SWITCH(csr.indptr->dtype, IdType, {
      ret = impl::CSRGetData<kDGLAscend, IdType, IdType>(
          csr, rows, cols, true, NDArray(), IdType(0));
    });
  } else
#endif
  {
    ATEN_CSR_SWITCH_CUDA_UVA(csr, rows, XPU, IdType, "CSRGetData", {
      ret = impl::CSRGetData<XPU, IdType>(csr, rows, cols);
    });
  }
  return ret;
}

template <typename DType>
NDArray CSRGetData(
    CSRMatrix csr, NDArray rows, NDArray cols, NDArray weights, DType filler) {
  NDArray ret;
  CHECK_SAME_DTYPE(csr.indices, rows);
  CHECK_SAME_DTYPE(csr.indices, cols);
  CHECK_SAME_CONTEXT(rows, cols);
  CHECK_SAME_CONTEXT(rows, weights);
#ifdef DGL_USE_ASCEND
  if (rows->ctx.device_type == kDGLAscend) {
    ATEN_ID_TYPE_SWITCH(csr.indptr->dtype, IdType, {
      ret = impl::CSRGetData<kDGLAscend, IdType, DType>(
          csr, rows, cols, false, weights, filler);
    });
  } else
#endif
  {
    ATEN_CSR_SWITCH_CUDA_UVA(csr, rows, XPU, IdType, "CSRGetData", {
      ret = impl::CSRGetData<XPU, IdType, DType>(
          csr, rows, cols, weights, filler);
    });
  }
  return ret;
}

runtime::NDArray CSRGetFloatingData(
    CSRMatrix csr, runtime::NDArray rows, runtime::NDArray cols,
    runtime::NDArray weights, double filler) {
  if (weights->dtype.bits == 64) {
    return CSRGetData<double>(csr, rows, cols, weights, filler);
  } else {
    CHECK(weights->dtype.bits == 32)
        << "CSRGetFloatingData only supports 32 or 64 bits floaring number";
    return CSRGetData<float>(csr, rows, cols, weights, filler);
  }
}

template NDArray CSRGetData<float>(
    CSRMatrix csr, NDArray rows, NDArray cols, NDArray weights, float filler);
template NDArray CSRGetData<double>(
    CSRMatrix csr, NDArray rows, NDArray cols, NDArray weights, double filler);

std::vector<NDArray> CSRGetDataAndIndices(
    CSRMatrix csr, NDArray rows, NDArray cols) {
  CHECK_SAME_DTYPE(csr.indices, rows);
  CHECK_SAME_DTYPE(csr.indices, cols);
  CHECK_SAME_CONTEXT(rows, cols);
  std::vector<NDArray> ret;
  if (rows->ctx.device_type == kDGLAscend) {
    DGLContext cpu_ctx{kDGLCPU, 0};
    auto csr_indptr = csr.indptr.CopyTo(cpu_ctx);
    auto csr_indices = csr.indices.CopyTo(cpu_ctx);
    auto csr_data = csr.data.CopyTo(cpu_ctx);
    auto rows_cpu = rows.CopyTo(cpu_ctx);
    auto cols_cpu = cols.CopyTo(cpu_ctx);
    CSRMatrix csr_cpu{csr.num_rows, csr.num_cols,
                       csr_indptr, csr_indices, csr_data};
    ATEN_ID_TYPE_SWITCH(csr.indptr->dtype, IdType, {
      ret = impl::CSRGetDataAndIndices<kDGLCPU, IdType>(
          csr_cpu, rows_cpu, cols_cpu);
    });
    for (auto& arr : ret) {
      arr = arr.CopyTo(rows->ctx);
    }
  } else {
    ATEN_CSR_SWITCH_CUDA_UVA(csr, rows, XPU, IdType, "CSRGetDataAndIndices", {
      ret = impl::CSRGetDataAndIndices<XPU, IdType>(csr, rows, cols);
    });
  }
  return ret;
}

CSRMatrix CSRTranspose(CSRMatrix csr) {
  CSRMatrix ret;
#ifdef DGL_USE_ASCEND
  if (csr.indptr->ctx.device_type == kDGLAscend) {
    ATEN_ID_TYPE_SWITCH(csr.indptr->dtype, IdType, {
      ret = impl::CSRTranspose<kDGLAscend, IdType>(csr);
    });
    return ret;
  }
#endif
  ATEN_XPU_SWITCH_CUDA(csr.indptr->ctx.device_type, XPU, "CSRTranspose", {
    ATEN_ID_TYPE_SWITCH(csr.indptr->dtype, IdType, {
      ret = impl::CSRTranspose<XPU, IdType>(csr);
    });
  });
  return ret;
}

COOMatrix CSRToCOO(CSRMatrix csr, bool data_as_order) {
  COOMatrix ret;
  if (data_as_order) {
    if (csr.indptr->ctx.device_type == kDGLAscend) {
      DGLContext cpu_ctx{kDGLCPU, 0};
      auto indptr_cpu = csr.indptr.CopyTo(cpu_ctx);
      auto indices_cpu = csr.indices.CopyTo(cpu_ctx);
      auto data_cpu = csr.data.CopyTo(cpu_ctx);
      CSRMatrix csr_cpu{csr.num_rows, csr.num_cols,
                         indptr_cpu, indices_cpu, data_cpu};
      ATEN_ID_TYPE_SWITCH(csr.indptr->dtype, IdType, {
        ret = impl::CSRToCOODataAsOrder<kDGLCPU, IdType>(csr_cpu);
      });
    } else {
      ATEN_XPU_SWITCH_CUDA(
          csr.indptr->ctx.device_type, XPU, "CSRToCOODataAsOrder", {
            ATEN_ID_TYPE_SWITCH(csr.indptr->dtype, IdType, {
              ret = impl::CSRToCOODataAsOrder<XPU, IdType>(csr);
            });
          });
    }
  } else {
#ifdef DGL_USE_ASCEND
    if (csr.indptr->ctx.device_type == kDGLAscend) {
      ATEN_ID_TYPE_SWITCH(csr.indptr->dtype, IdType, {
        ret = impl::CSRToCOO<kDGLAscend, IdType>(csr);
      });
    } else
#endif
    {
      ATEN_XPU_SWITCH_CUDA(csr.indptr->ctx.device_type, XPU, "CSRToCOO", {
        ATEN_ID_TYPE_SWITCH(csr.indptr->dtype, IdType, {
          ret = impl::CSRToCOO<XPU, IdType>(csr);
        });
      });
    }
  }
  return ret;
}

CSRMatrix CSRSliceRows(CSRMatrix csr, int64_t start, int64_t end) {
  CHECK(start >= 0 && start < csr.num_rows) << "Invalid start index: " << start;
  CHECK(end >= 0 && end <= csr.num_rows) << "Invalid end index: " << end;
  CHECK_GE(end, start);
  CSRMatrix ret;
#ifdef DGL_USE_ASCEND
  if (csr.indptr->ctx.device_type == kDGLAscend) {
    ATEN_ID_TYPE_SWITCH(csr.indptr->dtype, IdType, {
      ret = impl::CSRSliceRows<kDGLAscend, IdType>(csr, start, end);
    });
  } else
#endif
  {
    ATEN_CSR_SWITCH_CUDA(csr, XPU, IdType, "CSRSliceRows", {
      ret = impl::CSRSliceRows<XPU, IdType>(csr, start, end);
    });
  }
  return ret;
}

CSRMatrix CSRSliceRows(CSRMatrix csr, NDArray rows) {
  CHECK_SAME_DTYPE(csr.indices, rows);
  CSRMatrix ret;
#ifdef DGL_USE_ASCEND
  if (csr.indptr->ctx.device_type == kDGLAscend) {
    ATEN_ID_TYPE_SWITCH(csr.indptr->dtype, IdType, {
      ret = impl::CSRSliceRows<kDGLAscend, IdType>(csr, rows);
    });
  } else
#endif
  {
    ATEN_CSR_SWITCH_CUDA_UVA(csr, rows, XPU, IdType, "CSRSliceRows", {
      ret = impl::CSRSliceRows<XPU, IdType>(csr, rows);
    });
  }
  return ret;
}

CSRMatrix CSRSliceMatrix(CSRMatrix csr, NDArray rows, NDArray cols) {
  CHECK_SAME_DTYPE(csr.indices, rows);
  CHECK_SAME_DTYPE(csr.indices, cols);
  CHECK_SAME_CONTEXT(rows, cols);
  CSRMatrix ret;
#ifdef DGL_USE_ASCEND
  if (csr.indptr->ctx.device_type == kDGLAscend) {
    ATEN_ID_TYPE_SWITCH(csr.indptr->dtype, IdType, {
      ret = impl::CSRSliceMatrix<kDGLAscend, IdType>(csr, rows, cols);
    });
  } else
#endif
  {
    ATEN_CSR_SWITCH_CUDA_UVA(csr, rows, XPU, IdType, "CSRSliceMatrix", {
      ret = impl::CSRSliceMatrix<XPU, IdType>(csr, rows, cols);
    });
  }
  return ret;
}

void CSRSort_(CSRMatrix* csr) {
  if (csr->sorted) return;
  ATEN_CSR_SWITCH_CUDA(
      *csr, XPU, IdType, "CSRSort_", { impl::CSRSort_<XPU, IdType>(csr); });
}

std::pair<CSRMatrix, NDArray> CSRSortByTag(
    const CSRMatrix& csr, IdArray tag, int64_t num_tags) {
  CHECK_EQ(csr.indices->shape[0], tag->shape[0])
      << "The length of the tag array should be equal to the number of "
         "non-zero data.";
  CHECK_SAME_CONTEXT(csr.indices, tag);
  CHECK_INT(tag, "tag");
  std::pair<CSRMatrix, NDArray> ret;
  ATEN_CSR_SWITCH(csr, XPU, IdType, "CSRSortByTag", {
    ATEN_ID_TYPE_SWITCH(tag->dtype, TagType, {
      ret = impl::CSRSortByTag<XPU, IdType, TagType>(csr, tag, num_tags);
    });
  });
  return ret;
}

CSRMatrix CSRReorder(
    CSRMatrix csr, runtime::NDArray new_row_ids, runtime::NDArray new_col_ids) {
  CSRMatrix ret;
  ATEN_CSR_SWITCH(csr, XPU, IdType, "CSRReorder", {
    ret = impl::CSRReorder<XPU, IdType>(csr, new_row_ids, new_col_ids);
  });
  return ret;
}

CSRMatrix CSRRemove(CSRMatrix csr, IdArray entries) {
  CSRMatrix ret;
  ATEN_CSR_SWITCH(csr, XPU, IdType, "CSRRemove", {
    ret = impl::CSRRemove<XPU, IdType>(csr, entries);
  });
  return ret;
}

std::pair<COOMatrix, FloatArray> CSRLaborSampling(
    CSRMatrix mat, IdArray rows, int64_t num_samples, FloatArray prob,
    int importance_sampling, IdArray random_seed, float seed2_contribution,
    IdArray NIDs) {
  std::pair<COOMatrix, FloatArray> ret;
  ATEN_CSR_SWITCH_CUDA_UVA(mat, rows, XPU, IdType, "CSRLaborSampling", {
    const auto dtype =
        IsNullArray(prob) ? DGLDataTypeTraits<float>::dtype : prob->dtype;
    ATEN_FLOAT_TYPE_SWITCH(dtype, FloatType, "probability", {
      ret = impl::CSRLaborSampling<XPU, IdType, FloatType>(
          mat, rows, num_samples, prob, importance_sampling, random_seed,
          seed2_contribution, NIDs);
    });
  });
  return ret;
}

COOMatrix CSRRowWiseSampling(
    CSRMatrix mat, IdArray rows, int64_t num_samples, NDArray prob_or_mask,
    bool replace) {
  COOMatrix ret;
#ifdef DGL_USE_ASCEND
  if (mat.indptr->ctx.device_type == kDGLAscend) {
    if (IsNullArray(prob_or_mask)) {
      ATEN_ID_TYPE_SWITCH(mat.indptr->dtype, IdType, {
        ret = impl::CSRRowWiseSamplingUniform<kDGLAscend, IdType>(
            mat, rows, num_samples, replace);
      });
    } else {
      CHECK_VALID_CONTEXT(prob_or_mask, rows);
      // The AscendC weighted kernel operates in float32 (double is forbidden
      // in __aicore__). The launcher normalizes float64 / int8 / uint8
      // probability or mask to float32 on the host side; the sampling itself
      // runs natively on the NPU kernel.
      ATEN_ID_TYPE_SWITCH(mat.indptr->dtype, IdType, {
        ret = impl::CSRRowWiseSampling<kDGLAscend, IdType, float>(
            mat, rows, num_samples, prob_or_mask, replace);
      });
    }
  } else
#endif  // DGL_USE_ASCEND
  if (IsNullArray(prob_or_mask)) {
#ifdef DGL_USE_ASCEND
    if (rows->ctx.device_type == kDGLAscend) {
      ATEN_ID_TYPE_SWITCH(rows->dtype, IdType, {
        ret = impl::CSRRowWiseSamplingUniform<kDGLAscend, IdType>(
            mat, rows, num_samples, replace);
      });
    } else
#endif
    {
      ATEN_CSR_SWITCH_CUDA_UVA(
          mat, rows, XPU, IdType, "CSRRowWiseSamplingUniform", {
            ret = impl::CSRRowWiseSamplingUniform<XPU, IdType>(
                mat, rows, num_samples, replace);
          });
    }
  } else {
    // prob_or_mask is pinned and rows on GPU is valid
    CHECK_VALID_CONTEXT(prob_or_mask, rows);
    ATEN_CSR_SWITCH_CUDA_UVA(mat, rows, XPU, IdType, "CSRRowWiseSampling", {
      CHECK(!(prob_or_mask->dtype.bits == 8 && XPU == kDGLCUDA))
          << "GPU sampling with masks is currently not supported yet.";
      ATEN_FLOAT_INT8_UINT8_TYPE_SWITCH(
          prob_or_mask->dtype, FloatType, "probability or mask", {
            ret = impl::CSRRowWiseSampling<XPU, IdType, FloatType>(
                mat, rows, num_samples, prob_or_mask, replace);
          });
    });
  }
  return ret;
}

template <typename IdType, bool map_seed_nodes>
std::pair<CSRMatrix, IdArray> CSRRowWiseSamplingFused(
    CSRMatrix mat, IdArray rows, IdArray seed_mapping,
    std::vector<IdType>* new_seed_nodes, int64_t num_samples,
    NDArray prob_or_mask, bool replace) {
  std::pair<CSRMatrix, IdArray> ret;
#ifdef DGL_USE_ASCEND
  if (rows->ctx.device_type == kDGLAscend) {
    if (IsNullArray(prob_or_mask)) {
      ret = impl::CSRRowWiseSamplingUniformFused<
          kDGLAscend, IdType, map_seed_nodes>(
          mat, rows, seed_mapping, new_seed_nodes, num_samples, replace);
    } else {
      CHECK_VALID_CONTEXT(prob_or_mask, rows);
      // Weighted fused sampling has no CUDA reference either; explicit
      // named gap on NPU (see the fused operator docs in the process
      // repository). CHECK raises a catchable DGL error; LOG(FATAL)
      // would abort the process out from under Python.
      CHECK(false) << "Weighted fused sampling (prob/mask) is not supported "
                      "on NPU yet";
    }
  } else
#endif  // DGL_USE_ASCEND
  if (IsNullArray(prob_or_mask)) {
    ATEN_XPU_SWITCH(
        rows->ctx.device_type, XPU, "CSRRowWiseSamplingUniformFused", {
          ret =
              impl::CSRRowWiseSamplingUniformFused<XPU, IdType, map_seed_nodes>(
                  mat, rows, seed_mapping, new_seed_nodes, num_samples,
                  replace);
        });
  } else {
    CHECK_VALID_CONTEXT(prob_or_mask, rows);
    ATEN_XPU_SWITCH(rows->ctx.device_type, XPU, "CSRRowWiseSamplingFused", {
      ATEN_FLOAT_INT8_UINT8_TYPE_SWITCH(
          prob_or_mask->dtype, FloatType, "probability or mask", {
            ret = impl::CSRRowWiseSamplingFused<
                XPU, IdType, FloatType, map_seed_nodes>(
                mat, rows, seed_mapping, new_seed_nodes, num_samples,
                prob_or_mask, replace);
          });
    });
  }
  return ret;
}

template std::pair<CSRMatrix, IdArray> CSRRowWiseSamplingFused<int64_t, true>(
    CSRMatrix, IdArray, IdArray, std::vector<int64_t>*, int64_t, NDArray, bool);

template std::pair<CSRMatrix, IdArray> CSRRowWiseSamplingFused<int64_t, false>(
    CSRMatrix, IdArray, IdArray, std::vector<int64_t>*, int64_t, NDArray, bool);

template std::pair<CSRMatrix, IdArray> CSRRowWiseSamplingFused<int32_t, true>(
    CSRMatrix, IdArray, IdArray, std::vector<int32_t>*, int64_t, NDArray, bool);

template std::pair<CSRMatrix, IdArray> CSRRowWiseSamplingFused<int32_t, false>(
    CSRMatrix, IdArray, IdArray, std::vector<int32_t>*, int64_t, NDArray, bool);

COOMatrix CSRRowWisePerEtypeSampling(
    CSRMatrix mat, IdArray rows, const std::vector<int64_t>& eid2etype_offset,
    const std::vector<int64_t>& num_samples,
    const std::vector<NDArray>& prob_or_mask, bool replace,
    bool rowwise_etype_sorted) {
  COOMatrix ret;
  CHECK(prob_or_mask.size() > 0) << "probability or mask array is empty";
  ATEN_CSR_SWITCH(mat, XPU, IdType, "CSRRowWisePerEtypeSampling", {
    if (std::all_of(prob_or_mask.begin(), prob_or_mask.end(), IsNullArray)) {
      ret = impl::CSRRowWisePerEtypeSamplingUniform<XPU, IdType>(
          mat, rows, eid2etype_offset, num_samples, replace,
          rowwise_etype_sorted);
    } else {
      ATEN_FLOAT_INT8_UINT8_TYPE_SWITCH(
          prob_or_mask[0]->dtype, DType, "probability or mask", {
            ret = impl::CSRRowWisePerEtypeSampling<XPU, IdType, DType>(
                mat, rows, eid2etype_offset, num_samples, prob_or_mask, replace,
                rowwise_etype_sorted);
          });
    }
  });
  return ret;
}

COOMatrix CSRRowWiseTopk(
    CSRMatrix mat, IdArray rows, int64_t k, NDArray weight, bool ascending) {
  COOMatrix ret;
#ifdef DGL_USE_ASCEND
  if (mat.indptr->ctx.device_type == kDGLAscend) {
    CHECK_VALID_CONTEXT(weight, rows);
    // The AscendC sort network is float-only; the launcher normalizes
    // int32/int64/float64 weights to float32 (min-offset shift) before
    // launching the kernel.
    ATEN_ID_TYPE_SWITCH(mat.indptr->dtype, IdType, {
      ret = impl::CSRRowWiseTopk<kDGLAscend, IdType, float>(
          mat, rows, k, weight, ascending);
    });
    return ret;
  }
#endif  // DGL_USE_ASCEND
  ATEN_CSR_SWITCH(mat, XPU, IdType, "CSRRowWiseTopk", {
    ATEN_DTYPE_SWITCH(weight->dtype, DType, "weight", {
      ret = impl::CSRRowWiseTopk<XPU, IdType, DType>(
          mat, rows, k, weight, ascending);
    });
  });
  return ret;
}

COOMatrix CSRRowWiseSamplingBiased(
    CSRMatrix mat, IdArray rows, int64_t num_samples, NDArray tag_offset,
    FloatArray bias, bool replace) {
  COOMatrix ret;
#ifdef DGL_USE_ASCEND
  if (mat.indptr->ctx.device_type == kDGLAscend) {
    // The AscendC biased kernel operates in float32 (double is forbidden
    // in __aicore__). The launcher normalizes float64 bias to float32 on
    // the host side; the sampling itself runs natively on the NPU kernel.
    ATEN_ID_TYPE_SWITCH(mat.indptr->dtype, IdType, {
      ret = impl::CSRRowWiseSamplingBiased<kDGLAscend, IdType, float>(
          mat, rows, num_samples, tag_offset, bias, replace);
    });
    return ret;
  }
#endif  // DGL_USE_ASCEND
  ATEN_CSR_SWITCH(mat, XPU, IdType, "CSRRowWiseSamplingBiased", {
    ATEN_FLOAT_TYPE_SWITCH(bias->dtype, FloatType, "bias", {
      ret = impl::CSRRowWiseSamplingBiased<XPU, IdType, FloatType>(
          mat, rows, num_samples, tag_offset, bias, replace);
    });
  });
  return ret;
}

std::pair<IdArray, IdArray> CSRGlobalUniformNegativeSampling(
    const CSRMatrix& csr, int64_t num_samples, int num_trials,
    bool exclude_self_loops, bool replace, double redundancy) {
  CHECK_GT(num_samples, 0) << "Number of samples must be positive";
  CHECK_GT(num_trials, 0) << "Number of sampling trials must be positive";
  std::pair<IdArray, IdArray> result;
  ATEN_CSR_SWITCH_CUDA(csr, XPU, IdType, "CSRGlobalUniformNegativeSampling", {
    result = impl::CSRGlobalUniformNegativeSampling<XPU, IdType>(
        csr, num_samples, num_trials, exclude_self_loops, replace, redundancy);
  });
  return result;
}

CSRMatrix UnionCsr(const std::vector<CSRMatrix>& csrs) {
  CSRMatrix ret;
  CHECK_GT(csrs.size(), 1)
      << "UnionCsr creates a union of multiple CSRMatrixes";
  // sanity check
  for (size_t i = 1; i < csrs.size(); ++i) {
    CHECK_EQ(csrs[0].num_rows, csrs[i].num_rows)
        << "UnionCsr requires both CSRMatrix have same number of rows";
    CHECK_EQ(csrs[0].num_cols, csrs[i].num_cols)
        << "UnionCsr requires both CSRMatrix have same number of cols";
    CHECK_SAME_CONTEXT(csrs[0].indptr, csrs[i].indptr);
    CHECK_SAME_DTYPE(csrs[0].indptr, csrs[i].indptr);
  }

  ATEN_CSR_SWITCH(csrs[0], XPU, IdType, "UnionCsr", {
    ret = impl::UnionCsr<XPU, IdType>(csrs);
  });
  return ret;
}

std::tuple<CSRMatrix, IdArray, IdArray> CSRToSimple(const CSRMatrix& csr) {
  std::tuple<CSRMatrix, IdArray, IdArray> ret;

  CSRMatrix sorted_csr = (CSRIsSorted(csr)) ? csr : CSRSort(csr);
  ATEN_CSR_SWITCH(csr, XPU, IdType, "CSRToSimple", {
    ret = impl::CSRToSimple<XPU, IdType>(sorted_csr);
  });
  return ret;
}

///////////////////////// COO routines //////////////////////////

bool COOIsNonZero(COOMatrix coo, int64_t row, int64_t col) {
  bool ret = false;
  ATEN_COO_SWITCH(coo, XPU, IdType, "COOIsNonZero", {
    ret = impl::COOIsNonZero<XPU, IdType>(coo, row, col);
  });
  return ret;
}

NDArray COOIsNonZero(COOMatrix coo, NDArray row, NDArray col) {
  NDArray ret;
  ATEN_COO_SWITCH(coo, XPU, IdType, "COOIsNonZero", {
    ret = impl::COOIsNonZero<XPU, IdType>(coo, row, col);
  });
  return ret;
}

bool COOHasDuplicate(COOMatrix coo) {
  bool ret = false;
  ATEN_COO_SWITCH(coo, XPU, IdType, "COOHasDuplicate", {
    ret = impl::COOHasDuplicate<XPU, IdType>(coo);
  });
  return ret;
}

int64_t COOGetRowNNZ(COOMatrix coo, int64_t row) {
  int64_t ret = 0;
#ifdef DGL_USE_ASCEND
  if (coo.row->ctx.device_type == kDGLAscend) {
    ATEN_ID_TYPE_SWITCH(coo.row->dtype, IdType, {
      ret = impl::COOGetRowNNZ<kDGLAscend, IdType>(coo, row);
    });
    return ret;
  }
#endif
  {
    ATEN_COO_SWITCH_CUDA(coo, XPU, IdType, "COOGetRowNNZ", {
      ret = impl::COOGetRowNNZ<XPU, IdType>(coo, row);
    });
  }
  return ret;
}

NDArray COOGetRowNNZ(COOMatrix coo, NDArray row) {
  NDArray ret;
#ifdef DGL_USE_ASCEND
  if (coo.row->ctx.device_type == kDGLAscend) {
    ATEN_ID_TYPE_SWITCH(coo.row->dtype, IdType, {
      ret = impl::COOGetRowNNZ<kDGLAscend, IdType>(coo, row);
    });
    return ret;
  }
#endif
  {
    ATEN_COO_SWITCH_CUDA(coo, XPU, IdType, "COOGetRowNNZ", {
      ret = impl::COOGetRowNNZ<XPU, IdType>(coo, row);
    });
  }
  return ret;
}

std::pair<NDArray, NDArray> COOGetRowDataAndIndices(
    COOMatrix coo, int64_t row) {
  std::pair<NDArray, NDArray> ret;
#ifdef DGL_USE_ASCEND
  if (coo.row->ctx.device_type == kDGLAscend) {
    ATEN_ID_TYPE_SWITCH(coo.row->dtype, IdType, {
      ret = impl::COOGetRowDataAndIndices<kDGLAscend, IdType>(coo, row);
    });
    return ret;
  }
#endif
  {
    ATEN_COO_SWITCH(coo, XPU, IdType, "COOGetRowDataAndIndices", {
      ret = impl::COOGetRowDataAndIndices<XPU, IdType>(coo, row);
    });
  }
  return ret;
}

std::vector<NDArray> COOGetDataAndIndices(
    COOMatrix coo, NDArray rows, NDArray cols) {
  std::vector<NDArray> ret;
  if (coo.row->ctx.device_type == kDGLAscend) {
#ifdef DGL_USE_ASCEND
    aclrtSynchronizeDevice();  // Ensure PyTorch NPU ops complete before D2H copy
#endif
    DGLContext cpu_ctx{kDGLCPU, 0};
    auto coo_row = coo.row.CopyTo(cpu_ctx);
    auto coo_col = coo.col.CopyTo(cpu_ctx);
    auto coo_data = coo.data.CopyTo(cpu_ctx);
    auto rows_cpu = rows.CopyTo(cpu_ctx);
    auto cols_cpu = cols.CopyTo(cpu_ctx);
    COOMatrix coo_cpu{coo.num_rows, coo.num_cols,
                       coo_row, coo_col, coo_data};
    ATEN_ID_TYPE_SWITCH(coo.row->dtype, IdType, {
      ret = impl::COOGetDataAndIndices<kDGLCPU, IdType>(
          coo_cpu, rows_cpu, cols_cpu);
    });
    for (auto& arr : ret) {
      arr = arr.CopyTo(rows->ctx);
    }
  } else {
    ATEN_COO_SWITCH(coo, XPU, IdType, "COOGetDataAndIndices", {
      ret = impl::COOGetDataAndIndices<XPU, IdType>(coo, rows, cols);
    });
  }
  return ret;
}

NDArray COOGetData(COOMatrix coo, NDArray rows, NDArray cols) {
  NDArray ret;
  if (coo.row->ctx.device_type == kDGLAscend) {
#ifdef DGL_USE_ASCEND
    aclrtSynchronizeDevice();  // Ensure PyTorch NPU ops complete before D2H copy
#endif
    DGLContext cpu_ctx{kDGLCPU, 0};
    auto coo_row = coo.row.CopyTo(cpu_ctx);
    auto coo_col = coo.col.CopyTo(cpu_ctx);
    auto coo_data = coo.data.CopyTo(cpu_ctx);
    auto rows_cpu = rows.CopyTo(cpu_ctx);
    auto cols_cpu = cols.CopyTo(cpu_ctx);
    COOMatrix coo_cpu{coo.num_rows, coo.num_cols,
                       coo_row, coo_col, coo_data};
    ATEN_ID_TYPE_SWITCH(coo.row->dtype, IdType, {
      ret = impl::COOGetData<kDGLCPU, IdType>(coo_cpu, rows_cpu, cols_cpu);
    });
    ret = ret.CopyTo(rows->ctx);
  } else {
    ATEN_COO_SWITCH(coo, XPU, IdType, "COOGetData", {
      ret = impl::COOGetData<XPU, IdType>(coo, rows, cols);
    });
  }
  return ret;
}

COOMatrix COOTranspose(COOMatrix coo) {
  return COOMatrix(coo.num_cols, coo.num_rows, coo.col, coo.row, coo.data);
}

CSRMatrix COOToCSR(COOMatrix coo) {
  CSRMatrix ret;
#ifdef DGL_USE_ASCEND
  if (coo.row->ctx.device_type == kDGLAscend) {
    ATEN_ID_TYPE_SWITCH(
        coo.row->dtype, IdType, { ret = impl::COOToCSR<kDGLAscend, IdType>(coo); });
    return ret;
  }
#endif
  ATEN_XPU_SWITCH_CUDA(coo.row->ctx.device_type, XPU, "COOToCSR", {
    ATEN_ID_TYPE_SWITCH(
        coo.row->dtype, IdType, { ret = impl::COOToCSR<XPU, IdType>(coo); });
  });
  return ret;
}

COOMatrix COOSliceRows(COOMatrix coo, int64_t start, int64_t end) {
  COOMatrix ret;
#ifdef DGL_USE_ASCEND
  if (coo.row->ctx.device_type == kDGLAscend) {
    ATEN_ID_TYPE_SWITCH(coo.row->dtype, IdType, {
      ret = impl::COOSliceRows<kDGLAscend, IdType>(coo, start, end);
    });
    return ret;
  }
#endif
  ATEN_COO_SWITCH(coo, XPU, IdType, "COOSliceRows", {
    ret = impl::COOSliceRows<XPU, IdType>(coo, start, end);
  });
  return ret;
}

COOMatrix COOSliceRows(COOMatrix coo, NDArray rows) {
  COOMatrix ret;
#ifdef DGL_USE_ASCEND
  if (coo.row->ctx.device_type == kDGLAscend) {
    ATEN_ID_TYPE_SWITCH(coo.row->dtype, IdType, {
      ret = impl::COOSliceRows<kDGLAscend, IdType>(coo, rows);
    });
    return ret;
  }
#endif
  {
    ATEN_COO_SWITCH(coo, XPU, IdType, "COOSliceRows", {
      ret = impl::COOSliceRows<XPU, IdType>(coo, rows);
    });
  }
  return ret;
}

COOMatrix COOSliceMatrix(COOMatrix coo, NDArray rows, NDArray cols) {
  COOMatrix ret;
#ifdef DGL_USE_ASCEND
  if (coo.row->ctx.device_type == kDGLAscend) {
    ATEN_ID_TYPE_SWITCH(coo.row->dtype, IdType, {
      ret = impl::COOSliceMatrix<kDGLAscend, IdType>(coo, rows, cols);
    });
    return ret;
  }
#endif
  ATEN_COO_SWITCH(coo, XPU, IdType, "COOSliceMatrix", {
    ret = impl::COOSliceMatrix<XPU, IdType>(coo, rows, cols);
  });
  return ret;
}

void COOSort_(COOMatrix* mat, bool sort_column) {
  if ((mat->row_sorted && !sort_column) || mat->col_sorted) return;
#ifdef DGL_USE_ASCEND
  if (mat->row->ctx.device_type == kDGLAscend) {
    ATEN_ID_TYPE_SWITCH(mat->row->dtype, IdType, {
      impl::COOSort_<kDGLAscend, IdType>(mat, sort_column);
    });
    return;
  }
#endif
  {
    ATEN_XPU_SWITCH_CUDA(mat->row->ctx.device_type, XPU, "COOSort_", {
      ATEN_ID_TYPE_SWITCH(mat->row->dtype, IdType, {
        impl::COOSort_<XPU, IdType>(mat, sort_column);
      });
    });
  }
}

std::pair<bool, bool> COOIsSorted(COOMatrix coo) {
  if (coo.row->shape[0] <= 1) return {true, true};
  std::pair<bool, bool> ret;
  ATEN_COO_SWITCH_CUDA(coo, XPU, IdType, "COOIsSorted", {
    ret = impl::COOIsSorted<XPU, IdType>(coo);
  });
  return ret;
}

COOMatrix COOReorder(
    COOMatrix coo, runtime::NDArray new_row_ids, runtime::NDArray new_col_ids) {
  COOMatrix ret;
  ATEN_COO_SWITCH(coo, XPU, IdType, "COOReorder", {
    ret = impl::COOReorder<XPU, IdType>(coo, new_row_ids, new_col_ids);
  });
  return ret;
}

COOMatrix COORemove(COOMatrix coo, IdArray entries) {
  COOMatrix ret;
  ATEN_COO_SWITCH(coo, XPU, IdType, "COORemove", {
    ret = impl::COORemove<XPU, IdType>(coo, entries);
  });
  return ret;
}

std::pair<COOMatrix, FloatArray> COOLaborSampling(
    COOMatrix mat, IdArray rows, int64_t num_samples, FloatArray prob,
    int importance_sampling, IdArray random_seed, float seed2_contribution,
    IdArray NIDs) {
  std::pair<COOMatrix, FloatArray> ret;
  ATEN_COO_SWITCH(mat, XPU, IdType, "COOLaborSampling", {
    const auto dtype =
        IsNullArray(prob) ? DGLDataTypeTraits<float>::dtype : prob->dtype;
    ATEN_FLOAT_TYPE_SWITCH(dtype, FloatType, "probability", {
      ret = impl::COOLaborSampling<XPU, IdType, FloatType>(
          mat, rows, num_samples, prob, importance_sampling, random_seed,
          seed2_contribution, NIDs);
    });
  });
  return ret;
}

COOMatrix COORowWiseSampling(
    COOMatrix mat, IdArray rows, int64_t num_samples, NDArray prob_or_mask,
    bool replace) {
  COOMatrix ret;
#ifdef DGL_USE_ASCEND
  if (mat.row->ctx.device_type == kDGLAscend) {
    // Only the uniform path (no probability/mask) has an Ascend
    // implementation so far; reject the weighted path explicitly.
    CHECK(IsNullArray(prob_or_mask))
        << "COO weighted row-wise sampling on Ascend is not supported yet";
    ATEN_ID_TYPE_SWITCH(mat.row->dtype, IdType, {
      ret = impl::COORowWiseSamplingUniform<kDGLAscend, IdType>(
          mat, rows, num_samples, replace);
    });
    return ret;
  }
#endif  // DGL_USE_ASCEND
  ATEN_COO_SWITCH(mat, XPU, IdType, "COORowWiseSampling", {
    if (IsNullArray(prob_or_mask)) {
      ret = impl::COORowWiseSamplingUniform<XPU, IdType>(
          mat, rows, num_samples, replace);
    } else {
      ATEN_FLOAT_INT8_UINT8_TYPE_SWITCH(
          prob_or_mask->dtype, DType, "probability or mask", {
            ret = impl::COORowWiseSampling<XPU, IdType, DType>(
                mat, rows, num_samples, prob_or_mask, replace);
          });
    }
  });
  return ret;
}

COOMatrix COORowWisePerEtypeSampling(
    COOMatrix mat, IdArray rows, const std::vector<int64_t>& eid2etype_offset,
    const std::vector<int64_t>& num_samples,
    const std::vector<NDArray>& prob_or_mask, bool replace) {
  COOMatrix ret;
  CHECK(prob_or_mask.size() > 0) << "probability or mask array is empty";
  ATEN_COO_SWITCH(mat, XPU, IdType, "COORowWisePerEtypeSampling", {
    if (std::all_of(prob_or_mask.begin(), prob_or_mask.end(), IsNullArray)) {
      ret = impl::COORowWisePerEtypeSamplingUniform<XPU, IdType>(
          mat, rows, eid2etype_offset, num_samples, replace);
    } else {
      ATEN_FLOAT_INT8_UINT8_TYPE_SWITCH(
          prob_or_mask[0]->dtype, DType, "probability or mask", {
            ret = impl::COORowWisePerEtypeSampling<XPU, IdType, DType>(
                mat, rows, eid2etype_offset, num_samples, prob_or_mask,
                replace);
          });
    }
  });
  return ret;
}

COOMatrix COORowWiseTopk(
    COOMatrix mat, IdArray rows, int64_t k, FloatArray weight, bool ascending) {
  COOMatrix ret;
#ifdef DGL_USE_ASCEND
  if (mat.row->ctx.device_type == kDGLAscend) {
    CHECK_VALID_CONTEXT(weight, rows);
    // Assembly route (ADR-0001 pattern): the Ascend launcher converts the
    // full graph to CSR and runs the native CSR topk kernel; the weight
    // dtype normalization to f32 happens inside that kernel's launcher.
    ATEN_ID_TYPE_SWITCH(mat.row->dtype, IdType, {
      ret = impl::COORowWiseTopk<kDGLAscend, IdType, float>(
          mat, rows, k, weight, ascending);
    });
    return ret;
  }
#endif  // DGL_USE_ASCEND
  ATEN_COO_SWITCH(mat, XPU, IdType, "COORowWiseTopk", {
    ATEN_DTYPE_SWITCH(weight->dtype, DType, "weight", {
      ret = impl::COORowWiseTopk<XPU, IdType, DType>(
          mat, rows, k, weight, ascending);
    });
  });
  return ret;
}

std::pair<COOMatrix, IdArray> COOCoalesce(COOMatrix coo) {
  std::pair<COOMatrix, IdArray> ret;
  ATEN_COO_SWITCH(coo, XPU, IdType, "COOCoalesce", {
    ret = impl::COOCoalesce<XPU, IdType>(coo);
  });
  return ret;
}

COOMatrix DisjointUnionCoo(const std::vector<COOMatrix>& coos) {
  COOMatrix ret;
  ATEN_XPU_SWITCH_CUDA(coos[0].row->ctx.device_type, XPU, "DisjointUnionCoo", {
    ATEN_ID_TYPE_SWITCH(coos[0].row->dtype, IdType, {
      ret = impl::DisjointUnionCoo<XPU, IdType>(coos);
    });
  });
  return ret;
}

COOMatrix COOLineGraph(const COOMatrix& coo, bool backtracking) {
  COOMatrix ret;
  ATEN_COO_SWITCH(coo, XPU, IdType, "COOLineGraph", {
    ret = impl::COOLineGraph<XPU, IdType>(coo, backtracking);
  });
  return ret;
}

COOMatrix UnionCoo(const std::vector<COOMatrix>& coos) {
  COOMatrix ret;
  CHECK_GT(coos.size(), 1)
      << "UnionCoo creates a union of multiple COOMatrixes";
  // sanity check
  for (size_t i = 1; i < coos.size(); ++i) {
    CHECK_EQ(coos[0].num_rows, coos[i].num_rows)
        << "UnionCoo requires both COOMatrix have same number of rows";
    CHECK_EQ(coos[0].num_cols, coos[i].num_cols)
        << "UnionCoo requires both COOMatrix have same number of cols";
    CHECK_SAME_CONTEXT(coos[0].row, coos[i].row);
    CHECK_SAME_DTYPE(coos[0].row, coos[i].row);
  }

  // we assume the number of coos is not large in common cases
  std::vector<IdArray> coo_row;
  std::vector<IdArray> coo_col;
  bool has_data = false;

  for (size_t i = 0; i < coos.size(); ++i) {
    coo_row.push_back(coos[i].row);
    coo_col.push_back(coos[i].col);
    has_data |= COOHasData(coos[i]);
  }

  IdArray row = Concat(coo_row);
  IdArray col = Concat(coo_col);
  IdArray data = NullArray();

  if (has_data) {
    std::vector<IdArray> eid_data;
    eid_data.push_back(
        COOHasData(coos[0]) ? coos[0].data
                            : Range(
                                  0, coos[0].row->shape[0],
                                  coos[0].row->dtype.bits, coos[0].row->ctx));
    int64_t num_edges = coos[0].row->shape[0];
    for (size_t i = 1; i < coos.size(); ++i) {
      eid_data.push_back(
          COOHasData(coos[i])
              ? coos[i].data + num_edges
              : Range(
                    num_edges, num_edges + coos[i].row->shape[0],
                    coos[i].row->dtype.bits, coos[i].row->ctx));
      num_edges += coos[i].row->shape[0];
    }

    data = Concat(eid_data);
  }

  return COOMatrix(
      coos[0].num_rows, coos[0].num_cols, row, col, data, false, false);
}

std::tuple<COOMatrix, IdArray, IdArray> COOToSimple(const COOMatrix& coo) {
  // coo column sorted
  const COOMatrix sorted_coo = COOSort(coo, true);
  const IdArray eids_shuffled =
      COOHasData(sorted_coo)
          ? sorted_coo.data
          : Range(
                0, sorted_coo.row->shape[0], sorted_coo.row->dtype.bits,
                sorted_coo.row->ctx);
  const auto& coalesced_result = COOCoalesce(sorted_coo);
  const COOMatrix& coalesced_adj = coalesced_result.first;
  const IdArray& count = coalesced_result.second;

  /**
   * eids_shuffled actually already contains the mapping from old edge space to
   * the new one:
   *
   * * eids_shuffled[0:count[0]] indicates the original edge IDs that coalesced
   * into new edge #0.
   * * eids_shuffled[count[0]:count[0] + count[1]] indicates those that
   * coalesced into new edge #1.
   * * eids_shuffled[count[0] + count[1]:count[0] + count[1] + count[2]]
   * indicates those that coalesced into new edge #2.
   * * etc.
   *
   * Here, we need to translate eids_shuffled to an array "eids_remapped" such
   * that eids_remapped[i] indicates the new edge ID the old edge #i is mapped
   * to.  The translation can simply be achieved by (in numpy code):
   *
   *     new_eid_for_eids_shuffled = np.range(len(count)).repeat(count)
   *     eids_remapped = np.zeros_like(new_eid_for_eids_shuffled)
   *     eids_remapped[eids_shuffled] = new_eid_for_eids_shuffled
   */
  const IdArray new_eids = Range(
      0, coalesced_adj.row->shape[0], coalesced_adj.row->dtype.bits,
      coalesced_adj.row->ctx);
  const IdArray eids_remapped = Scatter(Repeat(new_eids, count), eids_shuffled);

  COOMatrix ret = COOMatrix(
      coalesced_adj.num_rows, coalesced_adj.num_cols, coalesced_adj.row,
      coalesced_adj.col, NullArray(), true, true);
  return std::make_tuple(ret, count, eids_remapped);
}

///////////////////////// Graph Traverse routines //////////////////////////
Frontiers BFSNodesFrontiers(const CSRMatrix& csr, IdArray source) {
  Frontiers ret;
  CHECK_EQ(csr.indptr->ctx.device_type, source->ctx.device_type)
      << "Graph and source should in the same device context";
  CHECK_EQ(csr.indices->dtype, source->dtype)
      << "Graph and source should in the same dtype";
  CHECK_EQ(csr.num_rows, csr.num_cols)
      << "Graph traversal can only work on square-shaped CSR.";
  ATEN_XPU_SWITCH_CUDA_ASCEND(source->ctx.device_type, XPU, "BFSNodesFrontiers", {
    ATEN_ID_TYPE_SWITCH(source->dtype, IdType, {
      ret = impl::BFSNodesFrontiers<XPU, IdType>(csr, source);
    });
  });
  return ret;
}

Frontiers BFSEdgesFrontiers(const CSRMatrix& csr, IdArray source) {
  Frontiers ret;
  CHECK_EQ(csr.indptr->ctx.device_type, source->ctx.device_type)
      << "Graph and source should in the same device context";
  CHECK_EQ(csr.indices->dtype, source->dtype)
      << "Graph and source should in the same dtype";
  CHECK_EQ(csr.num_rows, csr.num_cols)
      << "Graph traversal can only work on square-shaped CSR.";
  ATEN_XPU_SWITCH_CUDA_ASCEND(source->ctx.device_type, XPU, "BFSEdgesFrontiers", {
    ATEN_ID_TYPE_SWITCH(source->dtype, IdType, {
      ret = impl::BFSEdgesFrontiers<XPU, IdType>(csr, source);
    });
  });
  return ret;
}

Frontiers TopologicalNodesFrontiers(const CSRMatrix& csr) {
  Frontiers ret;
  CHECK_EQ(csr.num_rows, csr.num_cols)
      << "Graph traversal can only work on square-shaped CSR.";
  ATEN_XPU_SWITCH(
      csr.indptr->ctx.device_type, XPU, "TopologicalNodesFrontiers", {
        ATEN_ID_TYPE_SWITCH(csr.indices->dtype, IdType, {
          ret = impl::TopologicalNodesFrontiers<XPU, IdType>(csr);
        });
      });
  return ret;
}

Frontiers DGLDFSEdges(const CSRMatrix& csr, IdArray source) {
  Frontiers ret;
  CHECK_EQ(csr.indptr->ctx.device_type, source->ctx.device_type)
      << "Graph and source should in the same device context";
  CHECK_EQ(csr.indices->dtype, source->dtype)
      << "Graph and source should in the same dtype";
  CHECK_EQ(csr.num_rows, csr.num_cols)
      << "Graph traversal can only work on square-shaped CSR.";
  ATEN_XPU_SWITCH(source->ctx.device_type, XPU, "DGLDFSEdges", {
    ATEN_ID_TYPE_SWITCH(source->dtype, IdType, {
      ret = impl::DGLDFSEdges<XPU, IdType>(csr, source);
    });
  });
  return ret;
}

Frontiers DGLDFSLabeledEdges(
    const CSRMatrix& csr, IdArray source, const bool has_reverse_edge,
    const bool has_nontree_edge, const bool return_labels) {
  Frontiers ret;
  CHECK_EQ(csr.indptr->ctx.device_type, source->ctx.device_type)
      << "Graph and source should in the same device context";
  CHECK_EQ(csr.indices->dtype, source->dtype)
      << "Graph and source should in the same dtype";
  CHECK_EQ(csr.num_rows, csr.num_cols)
      << "Graph traversal can only work on square-shaped CSR.";
  ATEN_XPU_SWITCH(source->ctx.device_type, XPU, "DGLDFSLabeledEdges", {
    ATEN_ID_TYPE_SWITCH(source->dtype, IdType, {
      ret = impl::DGLDFSLabeledEdges<XPU, IdType>(
          csr, source, has_reverse_edge, has_nontree_edge, return_labels);
    });
  });
  return ret;
}

void CSRSpMM(
    const std::string& op, const std::string& reduce, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux) {
  const auto& bcast = CalcBcastOff(op, ufeat, efeat);

  ATEN_XPU_SWITCH_CUDA(csr.indptr->ctx.device_type, XPU, "SpMM", {
    ATEN_ID_TYPE_SWITCH(csr.indptr->dtype, IdType, {
      ATEN_FLOAT_TYPE_SWITCH_16BITS(out->dtype, Dtype, XPU, "Feature data", {
        SpMMCsr<XPU, IdType, Dtype>(
            op, reduce, bcast, csr, ufeat, efeat, out, out_aux);
      });
    });
  });
}

void CSRSpMM(
    const char* op, const char* reduce, const CSRMatrix& csr, NDArray ufeat,
    NDArray efeat, NDArray out, std::vector<NDArray> out_aux) {
  CSRSpMM(
      std::string(op), std::string(reduce), csr, ufeat, efeat, out, out_aux);
}

void CSRSDDMM(
    const std::string& op, const CSRMatrix& csr, NDArray ufeat, NDArray efeat,
    NDArray out, int lhs_target, int rhs_target) {
  const auto& bcast = CalcBcastOff(op, ufeat, efeat);

  ATEN_XPU_SWITCH_CUDA_ASCEND(csr.indptr->ctx.device_type, XPU, "SDDMM", {
    ATEN_ID_TYPE_SWITCH(csr.indptr->dtype, IdType, {
      ATEN_FLOAT_TYPE_SWITCH_16BITS(out->dtype, Dtype, XPU, "Feature data", {
        SDDMMCsr<XPU, IdType, Dtype>(
            op, bcast, csr, ufeat, efeat, out, lhs_target, rhs_target);
      });
    });
  });
}

void CSRSDDMM(
    const char* op, const CSRMatrix& csr, NDArray ufeat, NDArray efeat,
    NDArray out, int lhs_target, int rhs_target) {
  return CSRSDDMM(
      std::string(op), csr, ufeat, efeat, out, lhs_target, rhs_target);
}

void COOSpMM(
    const std::string& op, const std::string& reduce, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux) {
  const auto& bcast = CalcBcastOff(op, ufeat, efeat);

  ATEN_XPU_SWITCH_CUDA(coo.row->ctx.device_type, XPU, "SpMM", {
    ATEN_ID_TYPE_SWITCH(coo.row->dtype, IdType, {
      ATEN_FLOAT_TYPE_SWITCH_16BITS(out->dtype, Dtype, XPU, "Feature data", {
        SpMMCoo<XPU, IdType, Dtype>(
            op, reduce, bcast, coo, ufeat, efeat, out, out_aux);
      });
    });
  });
}

void COOSpMM(
    const char* op, const char* reduce, const COOMatrix& coo, NDArray ufeat,
    NDArray efeat, NDArray out, std::vector<NDArray> out_aux) {
  COOSpMM(
      std::string(op), std::string(reduce), coo, ufeat, efeat, out, out_aux);
}

void COOSDDMM(
    const std::string& op, const COOMatrix& coo, NDArray ufeat, NDArray efeat,
    NDArray out, int lhs_target, int rhs_target) {
  const auto& bcast = CalcBcastOff(op, ufeat, efeat);

  ATEN_XPU_SWITCH_CUDA_ASCEND(coo.row->ctx.device_type, XPU, "SDDMM", {
    ATEN_ID_TYPE_SWITCH(coo.row->dtype, IdType, {
      ATEN_FLOAT_TYPE_SWITCH_16BITS(out->dtype, Dtype, XPU, "Feature data", {
        SDDMMCoo<XPU, IdType, Dtype>(
            op, bcast, coo, ufeat, efeat, out, lhs_target, rhs_target);
      });
    });
  });
}

void COOSDDMM(
    const char* op, const COOMatrix& coo, NDArray ufeat, NDArray efeat,
    NDArray out, int lhs_target, int rhs_target) {
  COOSDDMM(std::string(op), coo, ufeat, efeat, out, lhs_target, rhs_target);
}

///////////////////////// C APIs /////////////////////////
DGL_REGISTER_GLOBAL("ndarray._CAPI_DGLConcatND")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      List<Value> arrays = args[0];
      const int64_t dim = args[1];
      *rv = ConcatND(ListValueToVector<NDArray>(arrays), dim);
    });

DGL_REGISTER_GLOBAL("ndarray._CAPI_DGLSparseMatrixGetFormat")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      SparseMatrixRef spmat = args[0];
      *rv = spmat->format;
    });

DGL_REGISTER_GLOBAL("ndarray._CAPI_DGLSparseMatrixGetNumRows")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      SparseMatrixRef spmat = args[0];
      *rv = spmat->num_rows;
    });

DGL_REGISTER_GLOBAL("ndarray._CAPI_DGLSparseMatrixGetNumCols")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      SparseMatrixRef spmat = args[0];
      *rv = spmat->num_cols;
    });

DGL_REGISTER_GLOBAL("ndarray._CAPI_DGLSparseMatrixGetIndices")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      SparseMatrixRef spmat = args[0];
      const int64_t i = args[1];
      *rv = spmat->indices[i];
    });

DGL_REGISTER_GLOBAL("ndarray._CAPI_DGLSparseMatrixGetFlags")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      SparseMatrixRef spmat = args[0];
      List<Value> flags;
      for (bool flg : spmat->flags) {
        flags.push_back(Value(MakeValue(flg)));
      }
      *rv = flags;
    });

DGL_REGISTER_GLOBAL("ndarray._CAPI_DGLCreateSparseMatrix")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const int32_t format = args[0];
      const int64_t nrows = args[1];
      const int64_t ncols = args[2];
      const List<Value> indices = args[3];
      const List<Value> flags = args[4];
      std::shared_ptr<SparseMatrix> spmat(new SparseMatrix(
          format, nrows, ncols, ListValueToVector<IdArray>(indices),
          ListValueToVector<bool>(flags)));
      *rv = SparseMatrixRef(spmat);
    });

DGL_REGISTER_GLOBAL("ndarray._CAPI_DGLExistSharedMemArray")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const std::string name = args[0];
#ifndef _WIN32
      *rv = SharedMemory::Exist(name);
#else
      *rv = false;
#endif  // _WIN32
    });

DGL_REGISTER_GLOBAL("ndarray._CAPI_DGLArrayCastToSigned")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      NDArray array = args[0];
      CHECK_EQ(array->dtype.code, kDGLUInt);
      std::vector<int64_t> shape(array->shape, array->shape + array->ndim);
      DGLDataType dtype = array->dtype;
      dtype.code = kDGLInt;
      *rv = array.CreateView(shape, dtype, 0);
    });

}  // namespace aten
}  // namespace dgl

std::ostream& operator<<(std::ostream& os, dgl::runtime::NDArray array) {
  return os << dgl::aten::ToDebugString(array);
}

