/**
 *  Copyright (c) 2018 by Contributors
 * @file scheduler/scheduler.cc
 * @brief DGL Scheduler implementation
 */
#include <dgl/scheduler.h>

#include <unordered_map>
#include <vector>

namespace dgl {
namespace sched {

template <class IdType>
std::vector<IdArray> DegreeBucketing(
    const IdArray& msg_ids, const IdArray& vids, const IdArray& recv_ids) {
  auto n_msgs = msg_ids->shape[0];

  const IdType* vid_data = static_cast<IdType*>(vids->data);
  const IdType* msg_id_data = static_cast<IdType*>(msg_ids->data);
  const IdType* recv_id_data = static_cast<IdType*>(recv_ids->data);

  // in edge: dst->msgs
  std::unordered_map<IdType, std::vector<IdType>> in_edges;
  for (IdType i = 0; i < n_msgs; ++i) {
    in_edges[vid_data[i]].push_back(msg_id_data[i]);
  }

  // bkt: deg->dsts
  std::unordered_map<IdType, std::vector<IdType>> bkt;
  for (const auto& it : in_edges) {
    bkt[it.second.size()].push_back(it.first);
  }

  std::unordered_set<IdType> zero_deg_nodes;
  for (IdType i = 0; i < recv_ids->shape[0]; ++i) {
    if (in_edges.find(recv_id_data[i]) == in_edges.end()) {
      zero_deg_nodes.insert(recv_id_data[i]);
    }
  }
  auto n_zero_deg = zero_deg_nodes.size();

  // calc output size
  IdType n_deg = bkt.size();
  IdType n_dst = in_edges.size();
  IdType n_mid_sec = bkt.size();  // zero deg won't affect message size
  if (n_zero_deg > 0) {
    n_deg += 1;
    n_dst += n_zero_deg;
  }

  // initialize output
  IdArray degs = IdArray::Empty({n_deg}, vids->dtype, vids->ctx);
  IdArray nids = IdArray::Empty({n_dst}, vids->dtype, vids->ctx);
  IdArray nid_section = IdArray::Empty({n_deg}, vids->dtype, vids->ctx);
  IdArray mids = IdArray::Empty({n_msgs}, vids->dtype, vids->ctx);
  IdArray mid_section = IdArray::Empty({n_mid_sec}, vids->dtype, vids->ctx);
  IdType* deg_ptr = static_cast<IdType*>(degs->data);
  IdType* nid_ptr = static_cast<IdType*>(nids->data);
  IdType* nsec_ptr = static_cast<IdType*>(nid_section->data);
  IdType* mid_ptr = static_cast<IdType*>(mids->data);
  IdType* msec_ptr = static_cast<IdType*>(mid_section->data);

  // fill in bucketing ordering
  for (const auto& it : bkt) {  // for each bucket
    const IdType deg = it.first;
    const IdType bucket_size = it.second.size();
    *deg_ptr++ = deg;
    *nsec_ptr++ = bucket_size;
    *msec_ptr++ = deg * bucket_size;
    for (const auto dst : it.second) {  // for each dst in this bucket
      *nid_ptr++ = dst;
      for (const auto mid : in_edges[dst]) {  // for each in edge of dst
        *mid_ptr++ = mid;
      }
    }
  }

  if (n_zero_deg > 0) {
    *deg_ptr = 0;
    *nsec_ptr = n_zero_deg;
    for (const auto dst : zero_deg_nodes) {
      *nid_ptr++ = dst;
    }
  }

  std::vector<IdArray> ret;
  ret.push_back(std::move(degs));
  ret.push_back(std::move(nids));
  ret.push_back(std::move(nid_section));
  ret.push_back(std::move(mids));
  ret.push_back(std::move(mid_section));

  return ret;
}

template std::vector<IdArray> DegreeBucketing<int32_t>(
    const IdArray& msg_ids, const IdArray& vids, const IdArray& recv_ids);

template std::vector<IdArray> DegreeBucketing<int64_t>(
    const IdArray& msg_ids, const IdArray& vids, const IdArray& recv_ids);

template <class IdType>
std::vector<IdArray> GroupEdgeByNodeDegree(
    const IdArray& uids, const IdArray& vids, const IdArray& eids) {
  auto n_edge = eids->shape[0];
  const IdType* eid_data = static_cast<IdType*>(eids->data);
  const IdType* uid_data = static_cast<IdType*>(uids->data);
  const IdType* vid_data = static_cast<IdType*>(vids->data);

  // node2edge: group_by nodes uid -> (eid, the other end vid)
  std::unordered_map<IdType, std::vector<std::pair<IdType, IdType>>> node2edge;
  for (IdType i = 0; i < n_edge; ++i) {
    node2edge[uid_data[i]].emplace_back(eid_data[i], vid_data[i]);
  }

  // bkt: deg -> group_by node uid
  std::unordered_map<IdType, std::vector<IdType>> bkt;
  for (const auto& it : node2edge) {
    bkt[it.second.size()].push_back(it.first);
  }

  // number of unique degree
  IdType n_deg = bkt.size();

  // initialize output
  IdArray degs = IdArray::Empty({n_deg}, eids->dtype, eids->ctx);
  IdArray new_uids = IdArray::Empty({n_edge}, uids->dtype, uids->ctx);
  IdArray new_vids = IdArray::Empty({n_edge}, vids->dtype, vids->ctx);
  IdArray new_eids = IdArray::Empty({n_edge}, eids->dtype, eids->ctx);
  IdArray sections = IdArray::Empty({n_deg}, eids->dtype, eids->ctx);
  IdType* deg_ptr = static_cast<IdType*>(degs->data);
  IdType* uid_ptr = static_cast<IdType*>(new_uids->data);
  IdType* vid_ptr = static_cast<IdType*>(new_vids->data);
  IdType* eid_ptr = static_cast<IdType*>(new_eids->data);
  IdType* sec_ptr = static_cast<IdType*>(sections->data);

  // fill in bucketing ordering
  for (const auto& it : bkt) {  // for each bucket
    // degree of this bucket
    const IdType deg = it.first;
    // number of edges in this bucket
    const IdType bucket_size = it.second.size();
    *deg_ptr++ = deg;
    *sec_ptr++ = deg * bucket_size;
    for (const auto u : it.second) {           // for uid in this bucket
      for (const auto& pair : node2edge[u]) {  // for each edge of uid
        *uid_ptr++ = u;
        *vid_ptr++ = pair.second;
        *eid_ptr++ = pair.first;
      }
    }
  }

  std::vector<IdArray> ret;
  ret.push_back(std::move(degs));
  ret.push_back(std::move(new_uids));
  ret.push_back(std::move(new_vids));
  ret.push_back(std::move(new_eids));
  ret.push_back(std::move(sections));

  return ret;
}

template std::vector<IdArray> GroupEdgeByNodeDegree<int32_t>(
    const IdArray& uids, const IdArray& vids, const IdArray& eids);

template std::vector<IdArray> GroupEdgeByNodeDegree<int64_t>(
    const IdArray& uids, const IdArray& vids, const IdArray& eids);

}  // namespace sched

}  // namespace dgl
