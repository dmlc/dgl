/*!
 *  Copyright (c) 2018 by Contributors
 * \file dgl/scheduler.h
 * \brief Operations on graph index.
 */
#ifndef DGL_SCHEDULER_H_
#define DGL_SCHEDULER_H_

#include <vector>
#include "runtime/ndarray.h"

namespace dgl {

typedef tvm::runtime::NDArray IdArray;

namespace sched {

/*!
 * \brief Generate degree bucketing schedule
 * \param vids The destination vertex for messages
 * \note If there are multiple messages going into the same destination vertex, then
 *       there will be multiple copies of the destination vertex in vids
 * \return a vector of 5 IdArrays for degree bucketing. The 5 arrays are:
 *         degrees: of degrees for each bucket
 *         nids: destination node ids
 *         nid_section: number of nodes in each bucket (used to split nids)
 *         mids: message ids
 *         mid_section: number of messages in each bucket (used to split mids)
 */
std::vector<IdArray> DegreeBucketing(const IdArray& vids);

}  // namespace sched

}  // namespace dgl

#endif  // DGL_SCHEDULER_H_
