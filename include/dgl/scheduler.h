// DGL Scheduler interface
#ifndef DGL_SCHEDULER_H_
#define DGL_SCHEDULER_H_

#include "runtime/ndarray.h"
#include <vector>

namespace dgl {

typedef tvm::runtime::NDArray IdArray;

namespace scheduler {

std::vector<IdArray> DegreeBucketing(const IdArray& vids);

} // namespace scheduler

} // namespace dgl

#endif // DGL_SCHEDULER_H_
