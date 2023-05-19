/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/include/csc_sampling_graph.h
 * @brief Header file of csc sampling graph.
 */

#include <torch/custom_class.h>
#include <torch/torch.h>

#include <string>
#include <vector>

namespace graphbolt {
namespace sampling {

using StringList = std::vector<std::string>;

/**
 * @brief Structure representing heterogeneous information about a graph.
 *
 * Example usage:
 *
 * Suppose the graph has 6 edges
 * node_offset = [0, 2, 4]
 * edge_types = [0, 1, 0, 2, 1, 2]
 * HeteroInfo info({"A", "B", "C"}, {"X", "Y", "Z"}, node_offset, edge_types);
 *
 * This example creates a `HeteroInfo` object with three node types ("A", "B",
 * "C") and three edge types ("X", "Y", "Z"). The `node_offset` tensor
 * represents the offset array of node type, the given array indicates that node
 * [0, 2) has type "A", [2, 4) has type "B", and [4, 6) has type "C". And the
 * `edge_types` tensor represents the type id of each edge.
 */
struct HeteroInfo {
  /**
   * @brief Constructs a new `HeteroInfo` object.
   * @param ntypes List of node types in the graph.
   * @param etypes List of edge types in the graph.
   * @param node_type_offset Offset array of node type. It is assumed that nodes
   * of same type have consecutive ids.
   * @param type_per_edge Type id of each edge, where type id is the
   * corresponding index of `edge_types`.
   */
  HeteroInfo(
      const StringList& ntypes, const StringList& etypes,
      torch::Tensor& node_type_offset, torch::Tensor& type_per_edge)
      : node_types(ntypes),
        edge_types(etypes),
        node_type_offset(node_type_offset),
        type_per_edge(type_per_edge) {}

  /** @brief Default constructor. */
  HeteroInfo() = default;

  /** @brief List of node types in the graph.*/
  StringList node_types;

  /** @brief List of edge types in the graph. */
  StringList edge_types;

  /**
   * @brief Offset array of node type. The length of it is equal to number of
   * node types.
   */
  torch::Tensor node_type_offset;

  /**
   * @brief Type id of each edge, where type id is the corresponding index of
   * edge_types. The length of it is equal to the number of edges.
   */
  torch::Tensor type_per_edge;

  /**
   * @brief Magic number to indicate Hetero info version in serialize/
   * deserialize stages.
   */
  static constexpr int64_t kHeteroInfoSerializeMagic = 0xDD2E60F0F6B4A129;

  /**
   * @brief Load hetero info from stream.
   * @param archive Input stream for deserializing.
   */
  void Load(torch::serialize::InputArchive& archive);

  /**
   * @brief Save hetero info to stream.
   * @param archive Output stream for serializing.
   */
  void Save(torch::serialize::OutputArchive& archive) const;
};

/**
 * @brief A sampling oriented csc format graph.
 */
class CSCSamplingGraph : public torch::CustomClassHolder {
 public:
  /**
   * @brief Constructor for CSC with data.
   * @param num_nodes The number of nodes in the graph.
   * @param indptr The CSC format index pointer array.
   * @param indices The CSC format index array.
   * @param hetero_info Heterogeneous graph information, if present. Nullptr
   * means it is a homogeneous graph.
   */
  CSCSamplingGraph(
      int64_t num_nodes, torch::Tensor& indptr, torch::Tensor& indices,
      const std::shared_ptr<HeteroInfo>& hetero_info);

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
   * @param ntypes A list of node types, if present.
   * @param etypes A list of edge types, if present.
   * @param node_type_offset_ A tensor representing the offset of node types, if
   * present.
   * @param type_per_edge_ A tensor representing the type of each edge, if
   * present.
   *
   * @return CSCSamplingGraph
   */
  static c10::intrusive_ptr<CSCSamplingGraph> FromCSCWithHeteroInfo(
      int64_t num_nodes, torch::Tensor indptr, torch::Tensor indices,
      const StringList& ntypes, const StringList& etypes,
      torch::Tensor node_type_offset, torch::Tensor type_per_edge);

  /** @brief Get the number of nodes. */
  int64_t NumNodes() const { return num_nodes_; }

  /** @brief Get the number of edges. */
  int64_t NumEdges() const { return indices_.size(0); }

  /** @brief Get the csc index pointer tensor. */
  const torch::Tensor CSCIndptr() const { return indptr_; }

  /** @brief Get the index tensor. */
  const torch::Tensor Indices() const { return indices_; }

  /** @brief Check if the graph is heterogeneous. */
  inline bool IsHeterogeneous() const { return hetero_info_ != nullptr; }

  /** @brief Get the node type offset tensor for a heterogeneous graph. */
  inline const torch::Tensor NodeTypeOffset() const {
    return hetero_info_->node_type_offset;
  }

  /** @brief Get the list of node types for a heterogeneous graph. */
  inline StringList& NodeTypes() const { return hetero_info_->node_types; }

  /** @brief Get the list of edge types for a heterogeneous graph. */
  inline const StringList& EdgeTypes() const {
    return hetero_info_->edge_types;
  }

  /** @brief Get the edge type tensor for a heterogeneous graph. */
  inline const torch::Tensor TypePerEdge() const {
    return hetero_info_->type_per_edge;
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
  /** @brief Heterogeneous graph information, if present. */
  std::shared_ptr<HeteroInfo> hetero_info_;
};

}  // namespace sampling
}  // namespace graphbolt

/**
 * @brief Overload stream operator to enable `torch::save()` and `torch.load()`
 * for CSCSamplingGraph.
 */
namespace torch {

/**
 * @brief Overload input stream operator for CSCSamplingGraph deserialization.
 * @param archive Input stream for deserializing.
 * @param graph CSCSamplingGraph.
 *
 * @return archive
 */
inline serialize::InputArchive& operator>>(
    serialize::InputArchive& archive,
    graphbolt::sampling::CSCSamplingGraph& graph);

/**
 * @brief Overload output stream operator for CSCSamplingGraph serialization.
 * @param archive Output stream for serializing.
 * @param graph CSCSamplingGraph.
 *
 * @return archive
 */
inline serialize::OutputArchive& operator<<(
    serialize::OutputArchive& archive,
    const graphbolt::sampling::CSCSamplingGraph& graph);

}  // namespace torch
