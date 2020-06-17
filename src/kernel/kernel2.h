/*!
 *  Copyright (c) 2020 by Contributors
 * \file kernel/kernel2.h
 * \brief New sparse kernel
 */
#ifndef DGL_KERNEL_KERNEL2_H_
#define DGL_KERNEL_KERNEL2_H_

#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include "./bcast.h"

namespace dgl {
namespace kernel {

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
           std::vector<NDArray> out_aux,
           SparseFormat format = SparseFormat::kAny);

template <int XPU, typename IdType, typename DType>
void SDDMMCsr(const std::string& op,
              const BcastOff& bcast,
              const aten::CSRMatrix& csr,
              NDArray ufeat,
              NDArray efeat,
              NDArray out,
              std::vector<NDArray> out_aux);

template <int XPU, typename IdType, typename DType>
void SDDMMCoo(const std::string& op,
              const BcastOff& bcast,
              const aten::COOMatrix& coo,
              NDArray ufeat,
              NDArray efeat,
              NDArray out,
              std::vector<NDArray> out_aux);

}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_KERNEL2_H_
