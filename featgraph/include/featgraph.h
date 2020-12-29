#ifndef FEATGRAPH_H_
#define FEATGRAPH_H_
#include <dlpack/dlpack.h>

namespace dgl {
namespace featgraph {

void SDDMMTreeReduction(DLManagedTensor* row, DLManagedTensor* col, 
                        DLManagedTensor* lhs, DLManagedTensor* rhs, 
                        DLManagedTensor* out);

}  // namespace featgraph
}  // namespace dgl

#endif  // FEATGRAPH_H_
