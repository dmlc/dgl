/*!
 *  Copyright (c) 2020 by Contributors
 * @file featgraph/include/featgraph.h
 * @brief FeatGraph kernel headers.
 */
#ifndef FEATGRAPH_H_
#define FEATGRAPH_H_

#include <dlpack/dlpack.h>

namespace dgl {
namespace featgraph {

/* @brief Load Featgraph module from given path. */
void LoadFeatGraphModule(const std::string& path);

/* @brief Call Featgraph's SDDMM kernel. */
void SDDMMTreeReduction(
    DLManagedTensor* row, DLManagedTensor* col, DLManagedTensor* lhs,
    DLManagedTensor* rhs, DLManagedTensor* out);

}  // namespace featgraph
}  // namespace dgl

#endif  // FEATGRAPH_H_
