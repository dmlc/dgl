/**
 *  Copyright (c) 2020 by Contributors
 * @file dgl/aten/array_ops.h
 * @brief Common array operations required by DGL.
 *
 * Note that this is not meant for a full support of array library such as ATen.
 * Only a limited set of operators required by DGL are implemented.
 */
#ifndef DGL_ATEN_ARRAY_OPS_H_
#define DGL_ATEN_ARRAY_OPS_H_

#include <algorithm>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "./types.h"

namespace dgl {
namespace aten {

//////////////////////////////////////////////////////////////////////
// ID array
//////////////////////////////////////////////////////////////////////

/** @return A special array to represent null. */
inline NDArray NullArray(
    const DGLDataType& dtype = DGLDataType{kDGLInt, 64, 1},
    const DGLContext& ctx = DGLContext{kDGLCPU, 0}) {
  return NDArray::Empty({0}, dtype, ctx);
}

/**
 * @return Whether the input array is a null array.
 */
inline bool IsNullArray(NDArray array) { return array->shape[0] == 0; }

/**
 * @brief Create a new id array with given length
 * @param length The array length
 * @param ctx The array context
 * @param nbits The number of integer bits
 * @return id array
 */
IdArray NewIdArray(
    int64_t length, DGLContext ctx = DGLContext{kDGLCPU, 0},
    uint8_t nbits = 64);

/**
 * @brief Create a new float array with given length
 * @param length The array length
 * @param ctx The array context
 * @param nbits The number of integer bits
 * @return float array
 */
FloatArray NewFloatArray(int64_t length,
                   DGLContext ctx = DGLContext{kDGLCPU, 0},
                   uint8_t nbits = 32);

/**
 * @brief Create a new id array using the given vector data
 * @param vec The vector data
 * @param nbits The integer bits of the returned array
 * @param ctx The array context
 * @return the id array
 */
template <typename T>
IdArray VecToIdArray(
    const std::vector<T>& vec, uint8_t nbits = 64,
    DGLContext ctx = DGLContext{kDGLCPU, 0});

/**
 * @brief Return an array representing a 1D range.
 * @param low Lower bound (inclusive).
 * @param high Higher bound (exclusive).
 * @param nbits result array's bits (32 or 64)
 * @param ctx Device context
 * @return range array
 */
IdArray Range(int64_t low, int64_t high, uint8_t nbits, DGLContext ctx);

/**
 * @brief Return an array full of the given value
 * @param val The value to fill.
 * @param length Number of elements.
 * @param nbits result array's bits (32 or 64)
 * @param ctx Device context
 * @return the result array
 */
IdArray Full(int64_t val, int64_t length, uint8_t nbits, DGLContext ctx);

/**
 * @brief Return an array full of the given value with the given type.
 * @param val The value to fill.
 * @param length Number of elements.
 * @param ctx Device context
 * @return the result array
 */
template <typename DType>
NDArray Full(DType val, int64_t length, DGLContext ctx);

/** @brief Create a deep copy of the given array */
IdArray Clone(IdArray arr);

/** @brief Convert the idarray to the given bit width */
IdArray AsNumBits(IdArray arr, uint8_t bits);

/** @brief Arithmetic functions */
IdArray Add(IdArray lhs, IdArray rhs);
IdArray Sub(IdArray lhs, IdArray rhs);
IdArray Mul(IdArray lhs, IdArray rhs);
IdArray Div(IdArray lhs, IdArray rhs);
IdArray Mod(IdArray lhs, IdArray rhs);

IdArray Add(IdArray lhs, int64_t rhs);
IdArray Sub(IdArray lhs, int64_t rhs);
IdArray Mul(IdArray lhs, int64_t rhs);
IdArray Div(IdArray lhs, int64_t rhs);
IdArray Mod(IdArray lhs, int64_t rhs);

IdArray Add(int64_t lhs, IdArray rhs);
IdArray Sub(int64_t lhs, IdArray rhs);
IdArray Mul(int64_t lhs, IdArray rhs);
IdArray Div(int64_t lhs, IdArray rhs);
IdArray Mod(int64_t lhs, IdArray rhs);

IdArray Neg(IdArray array);

// XXX(minjie): currently using integer array for bool type
IdArray GT(IdArray lhs, IdArray rhs);
IdArray LT(IdArray lhs, IdArray rhs);
IdArray GE(IdArray lhs, IdArray rhs);
IdArray LE(IdArray lhs, IdArray rhs);
IdArray EQ(IdArray lhs, IdArray rhs);
IdArray NE(IdArray lhs, IdArray rhs);

IdArray GT(IdArray lhs, int64_t rhs);
IdArray LT(IdArray lhs, int64_t rhs);
IdArray GE(IdArray lhs, int64_t rhs);
IdArray LE(IdArray lhs, int64_t rhs);
IdArray EQ(IdArray lhs, int64_t rhs);
IdArray NE(IdArray lhs, int64_t rhs);

IdArray GT(int64_t lhs, IdArray rhs);
IdArray LT(int64_t lhs, IdArray rhs);
IdArray GE(int64_t lhs, IdArray rhs);
IdArray LE(int64_t lhs, IdArray rhs);
IdArray EQ(int64_t lhs, IdArray rhs);
IdArray NE(int64_t lhs, IdArray rhs);

/** @brief Stack two arrays (of len L) into a 2*L length array */
IdArray HStack(IdArray arr1, IdArray arr2);

/** @brief Return the indices of the elements that are non-zero. */
IdArray NonZero(BoolArray bool_arr);

/**
 * @brief Return the data under the index. In numpy notation, A[I]
 * @tparam ValueType The type of return value.
 */
template <typename ValueType>
ValueType IndexSelect(NDArray array, int64_t index);

/**
 * @brief Return the data under the index. In numpy notation, A[I]
 */
NDArray IndexSelect(NDArray array, IdArray index);

/**
 * @brief Return the data from `start` (inclusive) to `end` (exclusive).
 */
NDArray IndexSelect(NDArray array, int64_t start, int64_t end);

/**
 * @brief Permute the elements of an array according to given indices.
 *
 * Only support 1D arrays.
 *
 * Equivalent to:
 *
 * <code>
 *     result = np.zeros_like(array)
 *     result[indices] = array
 * </code>
 */
NDArray Scatter(NDArray array, IdArray indices);

/**
 * @brief Scatter data into the output array.
 *
 * Equivalent to:
 *
 * <code>
 *     out[index] = value
 * </code>
 */
void Scatter_(IdArray index, NDArray value, NDArray out);

/**
 * @brief Repeat each element a number of times.  Equivalent to np.repeat(array,
 * repeats)
 * @param array A 1D vector
 * @param repeats A 1D integer vector for number of times to repeat for each
 * element in \c array.  Must have the same shape as \c array.
 */
NDArray Repeat(NDArray array, IdArray repeats);

/**
 * @brief Relabel the given ids to consecutive ids.
 *
 * Relabeling is done inplace. The mapping is created from the union
 * of the give arrays.
 *
 * Example:
 *
 * Given two IdArrays [2, 3, 10, 0, 2] and [4, 10, 5], one possible return
 * mapping is [2, 3, 10, 4, 0, 5], meaning the new ID 0 maps to the old ID
 * 2, 1 maps to 3, so on and so forth.
 *
 * @param arrays The id arrays to relabel.
 * @return mapping array M from new id to old id.
 */
IdArray Relabel_(const std::vector<IdArray>& arrays);

/**
 * @brief concatenate the given id arrays to one array
 *
 * Example:
 *
 * Given two IdArrays [2, 3, 10, 0, 2] and [4, 10, 5]
 * Return [2, 3, 10, 0, 2, 4, 10, 5]
 *
 * @param arrays The id arrays to concatenate.
 * @return concatenated array.
 */
NDArray Concat(const std::vector<IdArray>& arrays);

/** @brief Return whether the array is a valid 1D int array*/
inline bool IsValidIdArray(const dgl::runtime::NDArray& arr) {
  return arr->ndim == 1 && arr->dtype.code == kDGLInt;
}

/**
 * @brief Packs a tensor containing padded sequences of variable length.
 *
 * Similar to \c pack_padded_sequence in PyTorch, except that
 *
 * 1. The length for each sequence (before padding) is inferred as the number
 *    of elements before the first occurrence of \c pad_value.
 * 2. It does not sort the sequences by length.
 * 3. Along with the tensor containing the packed sequence, it returns both the
 *    length, as well as the offsets to the packed tensor, of each sequence.
 *
 * @param array The tensor containing sequences padded to the same length
 * @param pad_value The padding value
 * @return A triplet of packed tensor, the length tensor, and the offset tensor
 *
 * @note Example: consider the following array with padding value -1:
 *
 * <code>
 *     [[1, 2, -1, -1],
 *      [3, 4,  5, -1]]
 * </code>
 *
 * The packed tensor would be [1, 2, 3, 4, 5].
 *
 * The length tensor would be [2, 3], i.e. the length of each sequence before
 * padding.
 *
 * The offset tensor would be [0, 2], i.e. the offset to the packed tensor for
 * each sequence (before padding)
 */
template <typename ValueType>
std::tuple<NDArray, IdArray, IdArray> Pack(NDArray array, ValueType pad_value);

/**
 * @brief Batch-slice a 1D or 2D array, and then pack the list of sliced arrays
 * by concatenation.
 *
 * If a 2D array is given, then the function is equivalent to:
 *
 * <code>
 *     def ConcatSlices(array, lengths):
 *         slices = [array[i, :l] for i, l in enumerate(lengths)]
 *         packed = np.concatenate(slices)
 *         offsets = np.cumsum([0] + lengths[:-1])
 *         return packed, offsets
 * </code>
 *
 * If a 1D array is given, then the function is equivalent to
 *
 * <code>
 *     def ConcatSlices(array, lengths):
 *         slices = [array[:l] for l in lengths]
 *         packed = np.concatenate(slices)
 *         offsets = np.cumsum([0] + lengths[:-1])
 *         return packed, offsets
 * </code>
 *
 * @param array A 1D or 2D tensor for slicing
 * @param lengths A 1D tensor indicating the number of elements to slice
 * @return The tensor with packed slices along with the offsets.
 */
std::pair<NDArray, IdArray> ConcatSlices(NDArray array, IdArray lengths);

/**
 * @brief Return the cumulative summation (or inclusive sum) of the input array.
 *
 * The first element out[0] is equal to the first element of the input array
 * array[0]. The rest elements are defined recursively, out[i] = out[i-1] +
 * array[i]. Hence, the result array length is the same as the input array
 * length.
 *
 * If prepend_zero is true, then the first element is zero and the result array
 * length is the input array length plus one. This is useful for creating
 * an indptr array over a count array.
 *
 * @param array The 1D input array.
 * @return Array after cumsum.
 */
IdArray CumSum(IdArray array, bool prepend_zero = false);

/**
 * @brief Return the nonzero index.
 *
 * Only support 1D array. The result index array is in int64.
 *
 * @param array The input array.
 * @return A 1D index array storing the positions of the non zero values.
 */
IdArray NonZero(NDArray array);

/**
 * @brief Sort the ID vector in ascending order.
 *
 * It performs both sort and arg_sort (returning the sorted index). The sorted
 * index is always in int64.
 *
 * @param array Input array.
 * @param num_bits The number of bits used in key comparison. For example, if
 * the data type of the input array is int32_t and `num_bits = 8`, it only uses
 * bits in index range [0, 8) for sorting. Setting it to a small value could
 *                 speed up the sorting if the underlying sorting algorithm is
 * radix sort (e.g., on GPU). Setting it to zero (default value) means using all
 * the bits for comparison. On CPU, it currently has no effect.
 * @return A pair of arrays: sorted values and sorted index to the original
 * position.
 */
std::pair<IdArray, IdArray> Sort(IdArray array, int num_bits = 0);

/**
 * @brief Return a string that prints out some debug information.
 */
std::string ToDebugString(NDArray array);

// inline implementations
template <typename T>
IdArray VecToIdArray(const std::vector<T>& vec, uint8_t nbits, DGLContext ctx) {
  IdArray ret = NewIdArray(vec.size(), DGLContext{kDGLCPU, 0}, nbits);
  if (nbits == 32) {
    std::copy(vec.begin(), vec.end(), static_cast<int32_t*>(ret->data));
  } else if (nbits == 64) {
    std::copy(vec.begin(), vec.end(), static_cast<int64_t*>(ret->data));
  } else {
    LOG(FATAL) << "Only int32 or int64 is supported.";
  }
  return ret.CopyTo(ctx);
}

/**
 * @brief Get the context of the first array, and check if the non-null arrays'
 * contexts are the same.
 */
inline DGLContext GetContextOf(const std::vector<IdArray>& arrays) {
  bool first = true;
  DGLContext result;
  for (auto& array : arrays) {
    if (first) {
      first = false;
      result = array->ctx;
    } else {
      CHECK_EQ(array->ctx, result)
          << "Context of the input arrays are different";
    }
  }
  return result;
}

}  // namespace aten
}  // namespace dgl

#endif  // DGL_ATEN_ARRAY_OPS_H_
