// DGL Scheduler implementation

#include <unordered_map>
#include <vector>
#include <dgl/scheduler.h>

namespace dgl {

namespace scheduler {

std::vector<IdArray> DegreeBucketing(const IdArray& vids) {
    const auto n_msgs = vids->shape[0];
    const int64_t* vid_data = static_cast<int64_t*>(vids->data);

    // inedge: dst->msgs
    std::unordered_map<int64_t, std::vector<int64_t>> in_edges;
    for (int64_t mid = 0; mid < n_msgs; ++mid) {
        in_edges[vid_data[mid]].push_back(mid);
    }

    // bkt: deg->dsts
    std::unordered_map<int64_t, std::vector<int64_t>> bkt;
    for (auto& it: in_edges) {
        bkt[it.second.size()].push_back(it.first);
    }

    // initialize output
    int64_t n_deg = bkt.size();
    int64_t n_dst = in_edges.size();
    IdArray degs = IdArray::Empty({n_deg}, vids->dtype, vids->ctx);
    IdArray nids = IdArray::Empty({n_dst}, vids->dtype, vids->ctx);
    IdArray nid_section = IdArray::Empty({n_deg}, vids->dtype, vids->ctx);
    IdArray mids = IdArray::Empty({n_msgs}, vids->dtype, vids->ctx);
    IdArray mid_section = IdArray::Empty({n_deg}, vids->dtype, vids->ctx);
    int64_t* deg_ptr = static_cast<int64_t*>(degs->data);
    int64_t* nid_ptr = static_cast<int64_t*>(nids->data);
    int64_t* nsec_ptr = static_cast<int64_t*>(nid_section->data);
    int64_t* mid_ptr = static_cast<int64_t*>(mids->data);
    int64_t* msec_ptr = static_cast<int64_t*>(mid_section->data);

    // fill in bucketing ordering
    for (auto& it: bkt) { // for each bucket
        int64_t deg = it.first;
        int64_t n_dst = it.second.size();
        *deg_ptr++ = deg;
        *nsec_ptr++ = n_dst;
        *msec_ptr++ = deg * n_dst;
        for (auto dst: it.second) { // for each dst in this bucket
            *nid_ptr++ = dst;
            for (auto mid: in_edges[dst]) { // for each in edge of dst
                *mid_ptr++ = mid;
            }
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

} // namespace scheduler

} // namespace dgl
