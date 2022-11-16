/**
 *  Copyright (c) 2021 by Contributors
 * @file graph/sampling/frequency_hashmap.cuh
 * @brief frequency hashmap - used to select top-k frequency edges of each node
 */

#ifndef DGL_GRAPH_SAMPLING_RANDOMWALKS_FREQUENCY_HASHMAP_CUH_
#define DGL_GRAPH_SAMPLING_RANDOMWALKS_FREQUENCY_HASHMAP_CUH_

#include <dgl/array.h>
#include <dgl/runtime/device_api.h>

#include <tuple>

namespace dgl {
namespace sampling {
namespace impl {

template <typename IdxType>
class DeviceEdgeHashmap {
 public:
  struct EdgeItem {
    IdxType src;
    IdxType cnt;
  };
  DeviceEdgeHashmap() = delete;
  DeviceEdgeHashmap(
      int64_t num_dst, int64_t num_items_each_dst, IdxType *dst_unique_edges,
      EdgeItem *edge_hashmap)
      : _num_dst(num_dst),
        _num_items_each_dst(num_items_each_dst),
        _dst_unique_edges(dst_unique_edges),
        _edge_hashmap(edge_hashmap) {}
  // return the old cnt of this edge
  inline __device__ IdxType
  InsertEdge(const IdxType &src, const IdxType &dst_idx);
  inline __device__ IdxType GetDstCount(const IdxType &dst_idx);
  inline __device__ IdxType
  GetEdgeCount(const IdxType &src, const IdxType &dst_idx);

 private:
  int64_t _num_dst;
  int64_t _num_items_each_dst;
  IdxType *_dst_unique_edges;
  EdgeItem *_edge_hashmap;

  inline __device__ IdxType EdgeHash(const IdxType &id) const {
    return id % _num_items_each_dst;
  }
};

template <typename IdxType>
class FrequencyHashmap {
 public:
  static constexpr int64_t kDefaultEdgeTableScale = 3;
  FrequencyHashmap() = delete;
  FrequencyHashmap(
      int64_t num_dst, int64_t num_items_each_dst, DGLContext ctx,
      cudaStream_t stream, int64_t edge_table_scale = kDefaultEdgeTableScale);
  ~FrequencyHashmap();
  using EdgeItem = typename DeviceEdgeHashmap<IdxType>::EdgeItem;
  std::tuple<IdArray, IdArray, IdArray> Topk(
      const IdxType *src_data, const IdxType *dst_data, DGLDataType dtype,
      const int64_t num_edges, const int64_t num_edges_per_node,
      const int64_t num_pick);

 private:
  DGLContext _ctx;
  cudaStream_t _stream;
  DeviceEdgeHashmap<IdxType> *_device_edge_hashmap;
  IdxType *_dst_unique_edges;
  EdgeItem *_edge_hashmap;
};

};  // namespace impl
};  // namespace sampling
};  // namespace dgl

#endif  // DGL_GRAPH_SAMPLING_RANDOMWALKS_FREQUENCY_HASHMAP_CUH_
