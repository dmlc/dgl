/*!
 *  Copyright (c) 2018 by Contributors
 * \file scheduler/scheduler.cc
 * \brief DGL Scheduler implementation
 */
#include <dgl/scheduler.h>
#include <unordered_map>
#include <vector>

namespace dgl {
namespace sched {

std::vector<IdArray> DegreeBucketing(const IdArray& msg_ids, const IdArray& vids,
        const IdArray& recv_ids) {
    auto n_msgs = msg_ids->shape[0];

    const int64_t* vid_data = static_cast<int64_t*>(vids->data);
    const int64_t* msg_id_data = static_cast<int64_t*>(msg_ids->data);
    const int64_t* recv_id_data = static_cast<int64_t*>(recv_ids->data);

    // in edge: dst->msgs
    std::unordered_map<int64_t, std::vector<int64_t>> in_edges;
    for (int64_t i = 0; i < n_msgs; ++i) {
        in_edges[vid_data[i]].push_back(msg_id_data[i]);
    }

    // bkt: deg->dsts
    std::unordered_map<int64_t, std::vector<int64_t>> bkt;
    for (const auto& it : in_edges) {
        bkt[it.second.size()].push_back(it.first);
    }

    std::unordered_set<int64_t> zero_deg_nodes;
    for (int64_t i = 0; i < recv_ids->shape[0]; ++i) {
        if (in_edges.find(recv_id_data[i]) == in_edges.end()) {
            zero_deg_nodes.insert(recv_id_data[i]);
        }
    }
    auto n_zero_deg = zero_deg_nodes.size();

    // calc output size
    int64_t n_deg = bkt.size();
    int64_t n_dst = in_edges.size();
    int64_t n_mid_sec = bkt.size();  // zero deg won't affect message size
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
    int64_t* deg_ptr = static_cast<int64_t*>(degs->data);
    int64_t* nid_ptr = static_cast<int64_t*>(nids->data);
    int64_t* nsec_ptr = static_cast<int64_t*>(nid_section->data);
    int64_t* mid_ptr = static_cast<int64_t*>(mids->data);
    int64_t* msec_ptr = static_cast<int64_t*>(mid_section->data);

    // fill in bucketing ordering
    for (const auto& it : bkt) {  // for each bucket
        const int64_t deg = it.first;
        const int64_t bucket_size = it.second.size();
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

    return std::move(ret);
}

std::vector<IdArray> GroupEdgeByNodeDegree(const IdArray& uids, const IdArray& vids,
        const IdArray& eids) {
    auto n_edge = eids->shape[0];
    const int64_t* eid_data = static_cast<int64_t*>(eids->data);
    const int64_t* uid_data = static_cast<int64_t*>(uids->data);
    const int64_t* vid_data = static_cast<int64_t*>(vids->data);

    // node2edge: group_by nodes uid -> (eid, the other end vid)
    std::unordered_map<int64_t,
        std::vector<std::pair<int64_t, int64_t>>> node2edge;
    for (int64_t i = 0; i < n_edge; ++i) {
        node2edge[uid_data[i]].emplace_back(eid_data[i], vid_data[i]);
    }

    // bkt: deg -> group_by node uid
    std::unordered_map<int64_t, std::vector<int64_t>> bkt;
    for (const auto& it : node2edge) {
        bkt[it.second.size()].push_back(it.first);
    }

    // number of unique degree
    int64_t n_deg = bkt.size();

    // initialize output
    IdArray degs = IdArray::Empty({n_deg}, eids->dtype, eids->ctx);
    IdArray new_uids = IdArray::Empty({n_edge}, uids->dtype, uids->ctx);
    IdArray new_vids = IdArray::Empty({n_edge}, vids->dtype, vids->ctx);
    IdArray new_eids = IdArray::Empty({n_edge}, eids->dtype, eids->ctx);
    IdArray sections = IdArray::Empty({n_deg}, eids->dtype, eids->ctx);
    int64_t* deg_ptr = static_cast<int64_t*>(degs->data);
    int64_t* uid_ptr = static_cast<int64_t*>(new_uids->data);
    int64_t* vid_ptr = static_cast<int64_t*>(new_vids->data);
    int64_t* eid_ptr = static_cast<int64_t*>(new_eids->data);
    int64_t* sec_ptr = static_cast<int64_t*>(sections->data);

    // fill in bucketing ordering
    for (const auto& it : bkt) {  // for each bucket
        // degree of this bucket
        const int64_t deg = it.first;
        // number of edges in this bucket
        const int64_t bucket_size = it.second.size();
        *deg_ptr++ = deg;
        *sec_ptr++ = deg * bucket_size;
        for (const auto u : it.second) {  // for uid in this bucket
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

    return std::move(ret);
}

}  // namespace sched

}  // namespace dgl
