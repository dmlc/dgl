/**
 *  Copyright (c) 2017 by Contributors
 * @file pack_args.h
 * @brief Utility to pack DGLArgs to other type-erased fution calling
 * convention.
 *
 *  Two type erased function signatures are supported.
 *   - cuda_style(void** args, int num_args);
 *      - Pack everything by address
 *   - metal_style(void** buffers, int num_buffers,
 *                 union_32bit args[N], int num_args);
 *      - Pack buffer by address, pack rest parameter into 32bit union buffer.
 */
#ifndef DGL_RUNTIME_PACK_ARGS_H_
#define DGL_RUNTIME_PACK_ARGS_H_

#include <dgl/runtime/c_runtime_api.h>
#include <dgl/runtime/packed_func.h>

#include <cstring>
#include <vector>

namespace dgl {
namespace runtime {
/**
 * @brief argument union type of 32bit.
 * Choose 32 bit because most GPU API do not work well with 64 bit.
 */
union ArgUnion {
  int32_t v_int32;
  uint32_t v_uint32;
  float v_float32;
};
/**
 * @brief Create a packed function from void addr types.
 *
 * @param f with signiture (DGLArgs args, DGLRetValue* rv, void* void_args)
 * @param arg_types The arguments type information.
 * @tparam F the function type
 *
 * @return The wrapped packed function.
 */
template <typename F>
inline PackedFunc PackFuncVoidAddr(
    F f, const std::vector<DGLDataType>& arg_types);
/**
 * @brief Create a packed function that from function only packs buffer
 * arguments.
 *
 * @param f with signiture (DGLArgs args, DGLRetValue* rv, ArgUnion* pack_args)
 * @param arg_types The arguments type information.
 * @tparam F the function type
 *
 * @return The wrapped packed function.
 */
template <typename F>
inline PackedFunc PackFuncNonBufferArg(
    F f, const std::vector<DGLDataType>& arg_types);
/**
 * @brief Create a packed function that from function that takes a packed
 * arguments.
 *
 * @param f with signature (DGLArgs args, DGLRetValue* rv, void* pack_args,
 * size_t nbytes)
 * @param arg_types The arguments that wish to get from
 * @tparam F the function type
 *
 * @return The wrapped packed function.
 */
template <typename F>
inline PackedFunc PackFuncPackedArg(
    F f, const std::vector<DGLDataType>& arg_types);
/**
 * @brief Extract number of buffer argument from the argument types.
 * @param arg_types The argument types.
 * @return number of buffer arguments
 */
inline size_t NumBufferArgs(const std::vector<DGLDataType>& arg_types);

// implementations details
namespace detail {
template <typename T, int kSize>
class TempArray {
 public:
  explicit TempArray(int size) {}
  T* data() { return data_; }

 private:
  T data_[kSize];
};
template <typename T>
class TempArray<T, 0> {
 public:
  explicit TempArray(int size) : data_(size) {}
  T* data() { return data_.data(); }

 private:
  std::vector<T> data_;
};

/** @brief conversion code used in void arg. */
enum ArgConvertCode {
  INT64_TO_INT64,
  INT64_TO_INT32,
  INT64_TO_UINT32,
  FLOAT64_TO_FLOAT32,
  FLOAT64_TO_FLOAT64,
  HANDLE_TO_HANDLE
};

inline ArgConvertCode GetArgConvertCode(DGLDataType t) {
  CHECK_EQ(t.lanes, 1U)
      << "Cannot pass vector type argument to devic function for now";
  if (t.code == kDGLInt) {
    if (t.bits == 64U) return INT64_TO_INT64;
    if (t.bits == 32U) return INT64_TO_INT32;
  } else if (t.code == kDGLUInt) {
    if (t.bits == 32U) return INT64_TO_UINT32;
  } else if (t.code == kDGLFloat) {
    if (t.bits == 64U) return FLOAT64_TO_FLOAT64;
    if (t.bits == 32U) return FLOAT64_TO_FLOAT32;
  } else if (t.code == kHandle) {
    return HANDLE_TO_HANDLE;
  }
  LOG(FATAL) << "Cannot handle " << t << " as device function argument";
  return HANDLE_TO_HANDLE;
}

template <int N, typename F>
inline PackedFunc PackFuncVoidAddr_(
    F f, const std::vector<ArgConvertCode>& codes) {
  int num_args = static_cast<int>(codes.size());
  auto ret = [f, codes, num_args](DGLArgs args, DGLRetValue* ret) {
    TempArray<void*, N> addr_(num_args);
    TempArray<ArgUnion, N> holder_(num_args);
    void** addr = addr_.data();
    ArgUnion* holder = holder_.data();
    for (int i = 0; i < num_args; ++i) {
      switch (codes[i]) {
        case INT64_TO_INT64:
        case FLOAT64_TO_FLOAT64:
        case HANDLE_TO_HANDLE: {
          addr[i] = (void*)&(args.values[i]);  // NOLINT(*)
          break;
        }
        case INT64_TO_INT32: {
          holder[i].v_int32 = static_cast<int32_t>(args.values[i].v_int64);
          addr[i] = &(holder[i]);
          break;
        }
        case INT64_TO_UINT32: {
          holder[i].v_uint32 = static_cast<uint32_t>(args.values[i].v_int64);
          addr[i] = &(holder[i]);
          break;
        }
        case FLOAT64_TO_FLOAT32: {
          holder[i].v_float32 = static_cast<float>(args.values[i].v_float64);
          addr[i] = &(holder[i]);
          break;
        }
      }
    }
    f(args, ret, addr);
  };
  return PackedFunc(ret);
}

template <int N, typename F>
inline PackedFunc PackFuncNonBufferArg_(
    F f, int base, const std::vector<ArgConvertCode>& codes) {
  int num_args = static_cast<int>(codes.size());
  auto ret = [f, codes, base, num_args](DGLArgs args, DGLRetValue* ret) {
    TempArray<ArgUnion, N> holder_(num_args);
    ArgUnion* holder = holder_.data();
    for (int i = 0; i < num_args; ++i) {
      switch (codes[i]) {
        case INT64_TO_INT64:
        case FLOAT64_TO_FLOAT64: {
          LOG(FATAL) << "Donot support 64bit argument to device function";
          break;
        }
        case INT64_TO_INT32: {
          holder[i].v_int32 =
              static_cast<int32_t>(args.values[base + i].v_int64);
          break;
        }
        case INT64_TO_UINT32: {
          holder[i].v_uint32 =
              static_cast<uint32_t>(args.values[base + i].v_int64);
          break;
        }
        case FLOAT64_TO_FLOAT32: {
          holder[i].v_float32 =
              static_cast<float>(args.values[base + i].v_float64);
          break;
        }
        case HANDLE_TO_HANDLE: {
          LOG(FATAL) << "not reached";
          break;
        }
      }
    }
    f(args, ret, holder);
  };
  return PackedFunc(ret);
}

template <int N, typename F>
inline PackedFunc PackFuncPackedArg_(
    F f, const std::vector<ArgConvertCode>& codes) {
  int num_args = static_cast<int>(codes.size());
  auto ret = [f, codes, num_args](DGLArgs args, DGLRetValue* ret) {
    TempArray<uint64_t, N> pack_(num_args);
    int32_t* pack = reinterpret_cast<int32_t*>(pack_.data());
    int32_t* ptr = pack;
    static_assert(sizeof(DGLValue) == 8, "invariant");
    static_assert(sizeof(void*) % sizeof(int32_t) == 0, "invariant");
    for (int i = 0; i < num_args; ++i) {
      switch (codes[i]) {
        case HANDLE_TO_HANDLE: {
          std::memcpy(ptr, &(args.values[i].v_handle), sizeof(void*));
          ptr += sizeof(void*) / sizeof(int32_t);
          break;
        }
        case INT64_TO_INT64:
        case FLOAT64_TO_FLOAT64: {
          std::memcpy(ptr, &args.values[i], sizeof(DGLValue));
          ptr += 2;
          break;
        }
        case INT64_TO_INT32: {
          *ptr = static_cast<int32_t>(args.values[i].v_int64);
          ++ptr;
          break;
        }
        case INT64_TO_UINT32: {
          *reinterpret_cast<uint32_t*>(ptr) =
              static_cast<uint32_t>(args.values[i].v_int64);
          ++ptr;
          break;
        }
        case FLOAT64_TO_FLOAT32: {
          *reinterpret_cast<float*>(ptr) =
              static_cast<float>(args.values[i].v_float64);
          ++ptr;
          break;
        }
        default: {
          LOG(FATAL) << "not reached";
          break;
        }
      }
    }
    f(args, ret, pack, (ptr - pack) * sizeof(int32_t));
  };
  return PackedFunc(ret);
}
}  // namespace detail

template <typename F>
inline PackedFunc PackFuncVoidAddr(
    F f, const std::vector<DGLDataType>& arg_types) {
  std::vector<detail::ArgConvertCode> codes(arg_types.size());
  for (size_t i = 0; i < arg_types.size(); ++i) {
    codes[i] = detail::GetArgConvertCode(arg_types[i]);
  }
  size_t num_void_args = arg_types.size();
  // specialization
  if (num_void_args <= 4) {
    return detail::PackFuncVoidAddr_<4>(f, codes);
  } else if (num_void_args <= 8) {
    return detail::PackFuncVoidAddr_<8>(f, codes);
  } else {
    return detail::PackFuncVoidAddr_<0>(f, codes);
  }
}

inline size_t NumBufferArgs(const std::vector<DGLDataType>& arg_types) {
  size_t base = arg_types.size();
  for (size_t i = 0; i < arg_types.size(); ++i) {
    if (arg_types[i].code != kHandle) {
      base = i;
      break;
    }
  }
  for (size_t i = base; i < arg_types.size(); ++i) {
    CHECK(arg_types[i].code != kHandle)
        << "Device function need to be organized";
  }
  return base;
}

template <typename F>
inline PackedFunc PackFuncNonBufferArg(
    F f, const std::vector<DGLDataType>& arg_types) {
  size_t num_buffer = NumBufferArgs(arg_types);
  std::vector<detail::ArgConvertCode> codes;
  for (size_t i = num_buffer; i < arg_types.size(); ++i) {
    codes.push_back(detail::GetArgConvertCode(arg_types[i]));
  }
  int base = static_cast<int>(num_buffer);
  size_t nargs = codes.size();
  // specialization
  if (nargs <= 4) {
    return detail::PackFuncNonBufferArg_<4>(f, base, codes);
  } else {
    return detail::PackFuncNonBufferArg_<0>(f, base, codes);
  }
}

template <typename F>
inline PackedFunc PackFuncPackedArg(
    F f, const std::vector<DGLDataType>& arg_types) {
  std::vector<detail::ArgConvertCode> codes;
  for (size_t i = 0; i < arg_types.size(); ++i) {
    codes.push_back(detail::GetArgConvertCode(arg_types[i]));
  }
  size_t nargs = codes.size();
  // specialization
  if (nargs <= 4) {
    return detail::PackFuncPackedArg_<4>(f, codes);
  } else {
    return detail::PackFuncPackedArg_<0>(f, codes);
  }
}
}  // namespace runtime
}  // namespace dgl
#endif  // DGL_RUNTIME_PACK_ARGS_H_
