/*!
 *  Copyright (c) 2020 by Contributors
 * \file dgl/aten/kernel.h
 * \brief Sparse matrix operators.
 */
#ifndef DGL_KERNEL_H_
#define DGL_KERNEL_H_

#include <string>
#include <vector>

#include "array.h"
#include "./bcast.h"
#include "./base_heterograph.h"

namespace dgl {
namespace aten {

void SpMM(const std::string& op, const std::string& reduce,
          HeteroGraphPtr graph,
          NDArray ufeat,
          NDArray efeat,
          NDArray out,
          std::vector<NDArray> out_aux,
          SparseFormat format = SparseFormat::kAny);

template <int XPU, typename IdType, typename DType>
void SpMMCsr(const std::string& op, const std::string& reduce,
             const BcastOff& bcast,
             const aten::CSRMatrix& csr,
             NDArray ufeat,
             NDArray efeat,
             NDArray out,
             std::vector<NDArray> out_aux);

template <int XPU, typename IdType, typename DType>
void SpMMCoo(const std::string& op, const std::string& reduce,
             const BcastOff& bcast,
             const aten::COOMatrix& coo,
             NDArray ufeat,
             NDArray efeat,
             NDArray out,
             std::vector<NDArray> out_aux);

void SDDMM(const std::string& op,
           HeteroGraphPtr graph,
           NDArray ufeat,
           NDArray efeat,
           NDArray out,
           SparseFormat format = SparseFormat::kAny);

template <int XPU, typename IdType, typename DType>
void SDDMMCsr(const std::string& op,
              const BcastOff& bcast,
              const aten::CSRMatrix& csr,
              NDArray ufeat,
              NDArray efeat,
              NDArray out);

template <int XPU, typename IdType, typename DType>
void SDDMMCoo(const std::string& op,
              const BcastOff& bcast,
              const aten::COOMatrix& coo,
              NDArray ufeat,
              NDArray efeat,
              NDArray out);

}  // namespace aten
}  // namespace dgl

#endif  // DGL_KERNEL_H_
