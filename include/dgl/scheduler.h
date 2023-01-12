/**
 *  Copyright (c) 2018 by Contributors
 * @file dgl/scheduler.h
 * @brief Operations on graph index.
 */
#ifndef DGL_SCHEDULER_H_
#define DGL_SCHEDULER_H_

#include <vector>

#include "runtime/ndarray.h"

namespace dgl {

typedef dgl::runtime::NDArray IdArray;

namespace sched {

/**
 * @brief Generate degree bucketing schedule
 * @tparam IdType Graph's index data type, can be int32_t or int64_t
 * @param msg_ids The edge id for each message
 * @param vids The destination vertex for each message
 * @param recv_ids The recv nodes (for checking zero degree nodes)
 * @note If there are multiple messages going into the same destination vertex,
 *       then there will be multiple copies of the destination vertex in vids.
 * @return a vector of 5 IdArrays for degree bucketing. The 5 arrays are:
 *         degrees: degrees for each bucket
 *         nids: destination node ids
 *         nid_section: number of nodes in each bucket (used to split nids)
 *         mids: message ids
 *         mid_section: number of messages in each bucket (used to split mids)
 */
template <class IdType>
std::vector<IdArray> DegreeBucketing(
    const IdArray& msg_ids, const IdArray& vids, const IdArray& recv_ids);

/**
 * @brief Generate degree bucketing schedule for group_apply edge
 * @tparam IdType Graph's index data type, can be int32_t or int64_t
 * @param uids One end vertex of edge by which edges are grouped
 * @param vids The other end vertex of edge
 * @param eids Edge ids
 * @note This function always generate group_apply schedule based on degrees of
 *       nodes in uids. Therefore, if group_apply by source nodes, then uids
 *       should be source. If group_apply by destination nodes, then uids
 *       should be destination.
 * @return a vector of 5 IdArrays for degree bucketing. The 5 arrays are:
 *         degrees: degrees for each bucket
 *         new_uids: uids reordered by degree bucket
 *         new_vids: vids reordered by degree bucket
 *         new_edis: eids reordered by degree bucket
 *         sections: number of edges in each degree bucket (used to partition
 *                   new_uids, new_vids, and new_eids)
 */
template <class IdType>
std::vector<IdArray> GroupEdgeByNodeDegree(
    const IdArray& uids, const IdArray& vids, const IdArray& eids);

}  // namespace sched

}  // namespace dgl

#endif  // DGL_SCHEDULER_H_
