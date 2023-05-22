/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/csc_sampling_graph.h
 * @brief Header file of csc sampling graph.
 */
#ifndef GRAPHBOLT_CSC_SAMPLING_GRAPH_H_
#define GRAPHBOLT_CSC_SAMPLING_GRAPH_H_

#include <torch/custom_class.h>
#include <torch/torch.h>

#include <string>
#include <vector>

namespace graphbolt {
namespace sampling {

/**
 * @brief A sampling oriented csc format graph.
 */
class CSCSamplingGraph : public torch::CustomClassHolder {
 public:
  /** @brief Default constructor. */
  CSCSamplingGraph() = default;

  /**
   * @brief Constructor for CSC with data.
   * @param num_nodes The number of nodes in the graph.
   * @param indptr The CSC format index pointer array.
   * @param indices The CSC format index array.
   * @param node_type_offset A tensor representing the offset of node types, if
   * present.
   * @param type_per_edge A tensor representing the type of each edge, if
   * present.
   */
  CSCSamplingGraph(
      int64_t num_nodes, torch::Tensor& indptr, torch::Tensor& indices,
      const torch::optional<torch::Tensor>& node_type_offset,
      const torch::optional<torch::Tensor>& type_per_edge);

  /**
   * @brief Create a homogeneous CSC graph from tensors of CSC format.
   * @param num_nodes The number of nodes in the graph.
   * @param indptr Index pointer array of the CSC.
   * @param indices Indices array of the CSC.
   *
   * @return CSCSamplingGraph
   */
  static c10::intrusive_ptr<CSCSamplingGraph> FromCSC(
      int64_t num_nodes, torch::Tensor indptr, torch::Tensor indices);

  /**
   * @brief Create a heterogeneous CSC graph from tensors of CSC format.
   * @param num_nodes The number of nodes in the graph.
   * @param indptr Index pointer array of the CSC.
   * @param indices Indices array of the CSC.
   * @param node_type_offset A tensor representing the offset of node types.
   * @param type_per_edge A tensor representing the type of each edge.
   *
   * @return CSCSamplingGraph
   */
  static c10::intrusive_ptr<CSCSamplingGraph> FromCSCWithHeteroInfo(
      int64_t num_nodes, torch::Tensor indptr, torch::Tensor indices,
      torch::Tensor node_type_offset, torch::Tensor type_per_edge);

  /** @brief Get the number of nodes. */
  int64_t NumNodes() const { return num_nodes_; }

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

 private:
  /** @brief The number of nodes of the graph. */
  int64_t num_nodes_ = 0;
  /** @brief CSC format index pointer array. */
  torch::Tensor indptr_;
  /** @brief CSC format index array. */
  torch::Tensor indices_;
  /**
   * @brief Offset array of node type. The length of it is equal to number of
   * node types.
   */
  torch::optional<torch::Tensor> node_type_offset_;
  /**
   * @brief Type id of each edge, where type id is the corresponding index of
   * edge types. The length of it is equal to the number of edges.
   */
  torch::optional<torch::Tensor> type_per_edge_;
};

}  // namespace sampling
}  // namespace graphbolt

#endif  // GRAPHBOLT_CSC_SAMPLING_GRAPH_H_
