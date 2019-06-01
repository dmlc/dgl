/*!
 *  Copyright (c) 2019 by Contributors
 * \file dgl/heterograph.h
 * \brief DGL heterograph index class.
 */
#ifndef DGL_HETEROGRAPH_H_
#define DGL_HETEROGRAPH_H_

#include "immutable_graph.h"

namespace dgl {

class HeteroGraph;
typedef std::shared_ptr<HeteroGraph> HeteroGraphPtr;

class HeteroGraph {
  // This is indexed by edge type.
  std::vector<ImmutableGraphPtr> _subgraphs;
  int _num_ntypes;
  int _num_etypes;
 public:
  typedef std::pair<int, dgl_id_t> gnode_id_t;
  typedef int node_type_t;

  HeteroGraph(const std::vector<ImmutableGraphPtr> &subgraphs, int num_ntypes, int num_etypes) {
    this->_subgraphs = subgraphs;
    this->_num_ntypes = num_ntypes;
    this->_num_etypes = num_etypes;
  }

  class RelationType {
    int _src_type;
    int _etype;
    int _dst_type;
    int _num_ntypes;
    int _num_etypes;

    RelationType(int src_type, int dst_type, int num_ntypes) {
      this->_src_type = src_type;
      this->_dst_type = dst_type;
      this->_num_ntypes = num_ntypes;
      this->_etype = -1;
      this->_num_etypes = -1;
    }

    RelationType(int src_type, int etype, int dst_type, int num_ntypes, int num_etypes) {
      this->_src_type = src_type;
      this->_etype = etype;
      this->_dst_type = dst_type;
      this->_num_ntypes = num_ntypes;
      this->_num_etypes = num_etypes;
    }

   public:
    int src_type() const {
      return this->_src_type;
    }

    int dst_type() const {
      return this->_dst_type;
    }

    int etype() const {
      return this->_etype;
    }

    int get_id() const {
    }

    friend class HeteroGraph;
  };

  RelationType create_rtype(int src_type, int dst_type) const {
    return RelationType(src_type, dst_type, _num_ntypes);
  }

  RelationType create_rtype(int src_type, int etype, int dst_type) const {
    return RelationType(src_type, etype, dst_type, _num_ntypes, _num_etypes);
  }

  HeteroGraphPtr Slice(const RelationType &r);
  HeteroGraphPtr Slice(const std::vector<RelationType> &rs);

  static void SampleNeighbors(const CSR &csr, dgl_id_t local_vid, int k,
                              std::vector<dgl_id_t> *sampled);

  void SampleInNeighbors(gnode_id_t vid, node_type_t ntype, int k,
                         std::vector<dgl_id_t> *sampled) {
    auto g = _subgraphs[create_rtype(vid.first, ntype).get_id()];
    auto csr = g->GetInCSR();
    SampleNeighbors(*csr, vid.second, k, sampled);
  }

  void SampleInNeighbors(gnode_id_t vid, RelationType etype, int k,
                         std::vector<dgl_id_t> *sampled) {
    auto g = _subgraphs[etype.get_id()];
    auto csr = g->GetInCSR();
    SampleNeighbors(*csr, vid.second, k, sampled);
  }

  void SampleOutNeighbors(gnode_id_t vid, node_type_t ntype, int k,
                          std::vector<dgl_id_t> *sampled) {
    auto g = _subgraphs[create_rtype(vid.first, ntype).get_id()];
    auto csr = g->GetOutCSR();
    SampleNeighbors(*csr, vid.second, k, sampled);
  }

  void SampleOutNeighbors(gnode_id_t vid, RelationType etype, int k,
                          std::vector<dgl_id_t> *sampled) {
    auto g = _subgraphs[etype.get_id()];
    auto csr = g->GetOutCSR();
    SampleNeighbors(*csr, vid.second, k, sampled);
  }
};

}  // namespace dgl

#endif  // DGL_HETEROGRAPH_H_
