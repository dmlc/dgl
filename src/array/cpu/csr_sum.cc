#include <dgl/array.h>
#include <dgl/runtime/parallel_for.h>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>
#include <vector>
#include <stdexcept>
#include "array_utils.h"

namespace dgl {
namespace aten {

class CSRSumException : public std::runtime_error {
public:
    explicit CSRSumException(const std::string& message) : std::runtime_error(message) {}
};

template <typename IdType>
void CountNNZPerRow(const std::vector<const IdType*>& A_indptr,
                    const std::vector<const IdType*>& A_indices,
                    IdType* C_indptr_data, int64_t M) {
    int64_t n = A_indptr.size();

    runtime::parallel_for(0, M, [=](size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) {
            tsl::robin_set<IdType> set;
            for (int64_t k = 0; k < n; ++k) {
                for (IdType u = A_indptr[k][i]; u < A_indptr[k][i + 1]; ++u) {
                    set.insert(A_indices[k][u]);
                }
            }
            C_indptr_data[i] = set.size();
        }
    });
}

template <typename IdType>
int64_t ComputeIndptrInPlace(IdType* C_indptr_data, int64_t M) {
    int64_t nnz = 0;
    IdType len = 0;
    for (IdType i = 0; i < M; ++i) {
        len = C_indptr_data[i];
        C_indptr_data[i] = nnz;
        nnz += len;
    }
    C_indptr_data[M] = nnz;
    return nnz;
}

template <typename IdType, typename DType>
void ComputeIndicesAndData(const std::vector<const IdType*>& A_indptr,
                           const std::vector<const IdType*>& A_indices,
                           const std::vector<const IdType*>& A_eids,
                           const std::vector<const DType*>& A_data,
                           const IdType* C_indptr_data,
                           IdType* C_indices_data,
                           DType* C_weights_data, int64_t M) {
    int64_t n = A_indptr.size();
    runtime::parallel_for(0, M, [=](size_t b, size_t e) {
        for (auto i = b; i < e; ++i) {
            tsl::robin_map<IdType, DType> map;
            for (int64_t k = 0; k < n; ++k) {
                for (IdType u = A_indptr[k][i]; u < A_indptr[k][i + 1]; ++u) {
                    IdType kA = A_indices[k][u];
                    DType vA = A_data[k][A_eids[k] ? A_eids[k][u] : u];
                    map[kA] += vA;
                }
            }
            IdType j = C_indptr_data[i];
            for (auto it : map) {
                C_indices_data[j] = it.first;
                C_weights_data[j] = it.second;
                ++j;
            }
        }
    });
}

template <int XPU, typename IdType, typename DType>
std::pair<CSRMatrix, NDArray> CSRSum(const std::vector<CSRMatrix>& A,
                                     const std::vector<NDArray>& A_weights) {
    // Input validation
    if (A.empty()) {
        throw CSRSumException("List of matrices can't be empty.");
    }
    if (A.size() != A_weights.size()) {
        throw CSRSumException("List of matrices and weights must have same length.");
    }

    const int64_t M = A[0].num_rows;
    const int64_t N = A[0].num_cols;
    const int64_t n = A.size();

    // Validate consistency of matrix dimensions
    for (size_t i = 1; i < n; ++i) {
        if (A[i].num_rows != M || A[i].num_cols != N) {
            throw CSRSumException("All matrices must have the same dimensions.");
        }
    }

    std::vector<bool> A_has_eid(n);
    std::vector<const IdType*> A_indptr(n);
    std::vector<const IdType*> A_indices(n);
    std::vector<const IdType*> A_eids(n);
    std::vector<const DType*> A_data(n);

    for (int64_t i = 0; i < n; ++i) {
        const CSRMatrix& csr = A[i];
        const NDArray& data = A_weights[i];

        // Validate data types and contexts
        if (csr.indptr->dtype != A[0].indptr->dtype ||
            csr.indices->dtype != A[0].indices->dtype ||
            data->dtype != A_weights[0]->dtype ||
            csr.indptr->ctx != A[0].indptr->ctx ||
            csr.indices->ctx != A[0].indices->ctx ||
            data->ctx != A_weights[0]->ctx) {
            throw CSRSumException("Inconsistent data types or contexts across matrices.");
        }

        A_has_eid[i] = !IsNullArray(csr.data);
        A_indptr[i] = csr.indptr.Ptr<IdType>();
        A_indices[i] = csr.indices.Ptr<IdType>();
        A_eids[i] = A_has_eid[i] ? csr.data.Ptr<IdType>() : nullptr;
        A_data[i] = data.Ptr<DType>();

        // Validate array sizes
        if (csr.indptr->shape[0] != M + 1) {
            throw CSRSumException("Invalid indptr size for matrix " + std::to_string(i));
        }
        if (csr.indices->shape[0] != csr.indptr.Ptr<IdType>()[M]) {
            throw CSRSumException("Invalid indices size for matrix " + std::to_string(i));
        }
        if (data->shape[0] != csr.indices->shape[0]) {
            throw CSRSumException("Invalid data size for matrix " + std::to_string(i));
        }
    }

    IdArray C_indptr = IdArray::Empty({M + 1}, A[0].indptr->dtype, A[0].indptr->ctx);
    IdType* C_indptr_data = C_indptr.Ptr<IdType>();

    CountNNZPerRow<IdType>(A_indptr, A_indices, C_indptr_data, M);
    IdType nnz = ComputeIndptrInPlace<IdType>(C_indptr_data, M);

    // Allocate indices and weights array
    IdArray C_indices = IdArray::Empty({nnz}, A[0].indices->dtype, A[0].indices->ctx);
    NDArray C_weights = NDArray::Empty({nnz}, A_weights[0]->dtype, A_weights[0]->ctx);
    IdType* C_indices_data = C_indices.Ptr<IdType>();
    DType* C_weights_data = C_weights.Ptr<DType>();

    ComputeIndicesAndData<IdType, DType>(A_indptr, A_indices, A_eids, A_data,
                                         C_indptr_data, C_indices_data,
                                         C_weights_data, M);

    return {CSRMatrix(M, N, C_indptr, C_indices, NullArray(C_indptr->dtype, C_indptr->ctx)),
            C_weights};
}

// Explicit instantiations
template std::pair<CSRMatrix, NDArray> CSRSum<kDGLCPU, int32_t, float>(
    const std::vector<CSRMatrix>&, const std::vector<NDArray>&);
template std::pair<CSRMatrix, NDArray> CSRSum<kDGLCPU, int64_t, float>(
    const std::vector<CSRMatrix>&, const std::vector<NDArray>&);
template std::pair<CSRMatrix, NDArray> CSRSum<kDGLCPU, int32_t, double>(
    const std::vector<CSRMatrix>&, const std::vector<NDArray>&);
template std::pair<CSRMatrix, NDArray> CSRSum<kDGLCPU, int64_t, double>(
    const std::vector<CSRMatrix>&, const std::vector<NDArray>&);

} // namespace aten
} // namespace dgl