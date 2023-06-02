/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/csc_sampling_graph.h
 * @brief Header file of csc sampling graph.
 */
#ifndef GRAPHBOLT_CSC_SAMPLING_GRAPH_H_
#define GRAPHBOLT_CSC_SAMPLING_GRAPH_H_

#include <graphbolt/sampled_subgraph.h>
#include <graphbolt/shared_memory.h>
#include <torch/torch.h>

#include <string>
#include <vector>

namespace graphbolt {
namespace sampling {

/**
 * @brief A sampling oriented csc format graph.
 *
 * Example usage:
 *
 * Suppose the graph has 3 node types, 3 edge types and 6 edges
 * auto node_type_offset = {0, 2, 4, 6}
 * auto type_per_edge = {0, 1, 0, 2, 1, 2}
 * auto graph = CSCSamplingGraph(..., ..., node_type_offset, type_per_edge)
 *
 * The `node_type_offset` tensor represents the offset array of node type, the
 * given array indicates that node [0, 2) has type id 0, [2, 4) has type id 1,
 * and [4, 6) has type id 2. And the `type_per_edge` tensor represents the type
 * id of each edge.
 */
class CSCSamplingGraph : public torch::CustomClassHolder {
 public:
  /** @brief Default constructor. */
  CSCSamplingGraph() = default;

  /**
   * @brief Constructor for CSC with data.
   * @param indptr The CSC format index pointer array.
   * @param indices The CSC format index array.
   * @param node_type_offset A tensor representing the offset of node types, if
   * present.
   * @param type_per_edge A tensor representing the type of each edge, if
   * present.
   */
  CSCSamplingGraph(
      const torch::Tensor& indptr, const torch::Tensor& indices,
      const torch::optional<torch::Tensor>& node_type_offset,
      const torch::optional<torch::Tensor>& type_per_edge);

  /**
   * @brief Create a homogeneous CSC graph from tensors of CSC format.
   * @param indptr Index pointer array of the CSC.
   * @param indices Indices array of the CSC.
   * @param node_type_offset A tensor representing the offset of node types, if
   * present.
   * @param type_per_edge A tensor representing the type of each edge, if
   * present.
   *
   * @return CSCSamplingGraph
   */
  static c10::intrusive_ptr<CSCSamplingGraph> FromCSC(
      const torch::Tensor& indptr, const torch::Tensor& indices,
      const torch::optional<torch::Tensor>& node_type_offset,
      const torch::optional<torch::Tensor>& type_per_edge);

  /** @brief Get the number of nodes. */
  int64_t NumNodes() const { return indptr_.size(0) - 1; }

  /** @brief Get the number of edges. */
  int64_t NumEdges() const { return indices_.size(0); }

  /** @brief Get the csc index pointer tensor. */
  const torch::Tensor CSCIndptr() const { return indptr_; }

  /** @brief Get the index tensor. */
  const torch::Tensor Indices() const { return indices_; }

  /** @brief Get the node type offset tensor for a heterogeneous graph. */
  inline const torch::optional<torch::Tensor> NodeTypeOffset() const {
    return node_type_offset_;
  }

  /** @brief Get the edge type tensor for a heterogeneous graph. */
  inline const torch::optional<torch::Tensor> TypePerEdge() const {
    return type_per_edge_;
  }

  /**
   * @brief Magic number to indicate graph version in serialize/deserialize
   * stage.
   */
  static constexpr int64_t kCSCSamplingGraphSerializeMagic = 0xDD2E60F0F6B4A128;

  /**
   * @brief Load graph from stream.
   * @param archive Input stream for deserializing.
   */
  void Load(torch::serialize::InputArchive& archive);

  /**
   * @brief Save graph to stream.
   * @param archive Output stream for serializing.
   */
  void Save(torch::serialize::OutputArchive& archive) const;

  /**
   * @brief Return the subgraph induced on the inbound edges of the given nodes.
   * @param nodes Type agnostic node IDs to form the subgraph.
   *
   * @return SampledSubgraph.
   */
  c10::intrusive_ptr<SampledSubgraph> InSubgraph(
      const torch::Tensor& nodes) const;

  /**
   * @brief Sample neighboring edges of the given nodes and return the induced
   * subgraph.
   *
   * @param nodes The nodes from which to sample neighbors.
   * @param fanout The number of edges to be sampled for each node. It should be
   * >= 0 or -1. If -1 is given, all neighbors will be selected. Otherwise, it
   * will pick the minimum number of neighbors between the fanout value and the
   * total number of neighbors.
   * @param replace Boolean indicating whether the sample is preformed with or
   * without replacement. If True, a value can be selected multiple
   * times.Otherwise, each value can be selected only once.
   *
   * @return An intrusive pointer to a SampledSubgraph object containing the
   * sampled graph's information.
   */
  c10::intrusive_ptr<SampledSubgraph> SampleNeighbors(
      const torch::Tensor& nodes, int64_t fanout, bool replace) const;

  /**
   * @brief Copy the graph to shared memory.
   * @param shared_memory_name The name of the shared memory.
   *
   * @return A new CSCSamplingGraph object on shared memory.
   */
  c10::intrusive_ptr<CSCSamplingGraph> CopyToSharedMemory(
      const std::string& shared_memory_name);

  /**
   * @brief Load the graph from shared memory.
   * @param shared_memory_name The name of the shared memory.
   *
   * @return A new CSCSamplingGraph object on shared memory.
   */
  static c10::intrusive_ptr<CSCSamplingGraph> LoadFromSharedMemory(
      const std::string& shared_memory_name);

 private:
  /**
   * @brief Build a CSCSamplingGraph from shared memory tensors.
   *
   * @param shared_memory_tensors A tuple of two share memory objects holding
   * tensor meta information and data respectively, and a vector of optional
   * tensors on shared memory.
   *
   * @return A new CSCSamplingGraph on shared memory.
   */
  static c10::intrusive_ptr<CSCSamplingGraph> BuildGraphFromSharedMemoryTensors(
      std::tuple<
          SharedMemoryPtr, SharedMemoryPtr,
          std::vector<torch::optional<torch::Tensor>>>&& shared_memory_tensors);

  /** @brief CSC format index pointer array. */
  torch::Tensor indptr_;

  /** @brief CSC format index array. */
  torch::Tensor indices_;

  /**
   * @brief Offset array of node type. The length of it is equal to the number
   * of node types + 1. The tensor is in ascending order as nodes of the same
   * type have continuous IDs, and larger node IDs are paired with larger node
   * type IDs. Its first value is 0 and last value is the number of nodes. And
   * nodes with ID between `node_type_offset_[i] ~ node_type_offset_[i+1]` are
   * of type id `i`.
   */
  torch::optional<torch::Tensor> node_type_offset_;

  /**
   * @brief Type id of each edge, where type id is the corresponding index of
   * edge types. The length of it is equal to the number of edges.
   */
  torch::optional<torch::Tensor> type_per_edge_;

  /**
   * @brief Maximum number of bytes used to serialize the metadata of the
   * member tensors, including tensor shape and dtype. The constant is estimated
   * by multiplying the number of tensors in this class and the maximum number
   * of bytes used to serialize the metadata of a tensor (4 * 8192 for now).
   */
  static constexpr int64_t SERIALIZED_METAINFO_SIZE_MAX = 32768;

  /**
   * @brief Shared memory used to hold the tensor meta information and data of
   * this class. By storing its shared memory objects, the graph controls the
   * resources of shared memory, which will be released automatically when the
   * graph is destroyed.
   */
  SharedMemoryPtr tensor_meta_shm_, tensor_data_shm_;
};

/**
 * @brief Picks a specified number of neighbors for a node, starting from the
 * given offset and having the specified number of neighbors.
 *
 * @param offset The starting edge ID for the connected neighbors of the sampled
 * node.
 * @param num_neighbors The number of neighbors to pick.
 * @param fanout The number of edges to be sampled for each node. It should be
 * >= 0 or -1. If -1 is given, all neighbors will be selected. Otherwise, it
 * will pick the minimum number of neighbors between the fanout value and the
 * total number of neighbors.
 * @param replace Boolean indicating whether the sample is preformed with or
 * without replacement. If True, a value can be selected multiple
 * times.Otherwise, each value can be selected only once.
 * @param options Tensor options specifying the desired data type of the result.
 *
 * @return A tensor containing the picked neighbors.
 */
torch::Tensor Pick(
    int64_t offset, int64_t num_neighbors, int64_t fanout, bool replace,
    const torch::TensorOptions& options);

}  // namespace sampling
}  // namespace graphbolt

#endif  // GRAPHBOLT_CSC_SAMPLING_GRAPH_H_
