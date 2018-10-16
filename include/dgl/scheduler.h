// DGL Scheduler interface
#ifndef DGL_SCHEDULER_H_
#define DGL_SCHEDULER_H_

#include "runtime/ndarray.h"
#include <vector>

namespace dgl {

typedef tvm::runtime::NDArray IdArray;

namespace scheduler {

/*!
 * \brief Generate degree bucketing schedule
 * \param vids The destination vertex for messages
 * \note If there are multiple messages going into the same destination vertex, then
 *       there will be multiple copies of the destination vertex in vids
 * \return a vector of 5 IdArrays for degree bucketing
 */
std::vector<IdArray> DegreeBucketing(const IdArray& vids);

} // namespace scheduler

} // namespace dgl

#endif // DGL_SCHEDULER_H_
