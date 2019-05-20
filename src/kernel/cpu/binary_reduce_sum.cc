/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cpu/binary_reduce_sum.cc
 * \brief CPU kernels for binary reduce sum
 */
#include "./binary_reduce_impl.h"
#include "./backward_binary_reduce_impl.h"

namespace dgl {
namespace kernel {

#define REDUCER ReduceSum
#define XPU kDLCPU

//EVAL(GEN_DTYPE, GEN_TARGET, GEN_BINARY_OP, GEN_DEFINE)
//EVAL(GEN_BACKWARD_MODE, GEN_DTYPE, GEN_TARGET, GEN_BINARY_OP, GEN_BACKWARD_DEFINE)

template void CallBinaryReduce<XPU, float, SelectSrc, SelectDst,
         BinaryAdd<float>, REDUCER<XPU, float>>(
             const minigun::advance::RuntimeConfig& rtcfg,
             const minigun::Csr& csr,
             const minigun::Csr& rev_csr,
             GData<float>* gdata);
template void CallBinaryReduce<XPU, float, SelectSrc, SelectDst,
         BinarySub<float>, REDUCER<XPU, float>>(
             const minigun::advance::RuntimeConfig& rtcfg,
             const minigun::Csr& csr,
             const minigun::Csr& rev_csr,
             GData<float>* gdata);
template void CallBinaryReduce<XPU, float, SelectDst, SelectSrc,
         BinaryAdd<float>, REDUCER<XPU, float>>(
             const minigun::advance::RuntimeConfig& rtcfg,
             const minigun::Csr& csr,
             const minigun::Csr& rev_csr,
             GData<float>* gdata);
template void CallBinaryReduce<XPU, float, SelectDst, SelectSrc,
         BinarySub<float>, REDUCER<XPU, float>>(
             const minigun::advance::RuntimeConfig& rtcfg,
             const minigun::Csr& csr,
             const minigun::Csr& rev_csr,
             GData<float>* gdata);

}  // namespace kernel
}  // namespace dgl
