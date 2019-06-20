//
// Created by Allen Zhou on 2019/6/19.
//

#include "graph_serialize.h"

using dgl::COO;
using dgl::COOPtr;
using dgl::ImmutableGraph;
using dmlc::SeekStream;
using dgl::runtime::NDArray;

namespace dgl {
namespace serialize {

typedef std::pair<std::string, NDArray> NamedTensor;

const int kVersion = 1;

enum GraphType {
kMutableGraph = 0,
kImmutableGraph = 1
};

static void ToNameAndTensorList(void *pytensorlist, void *pynamelist, int list_size,
                                std::vector<std::string> &name_listptr,
                                std::vector<NDArray> &tensor_listptr) {
  const NDArray *tensor_list_handle = static_cast<const NDArray *>(pytensorlist);
  const std::string *name_list_handle = static_cast<const std::string *>(pynamelist);
  for (int i = 0; i < list_size; i ++) {
    tensor_listptr.push_back(tensor_list_handle[i]);
    name_listptr.push_back(name_list_handle[i]);
  }
}


DGL_REGISTER_GLOBAL("graph_serialize._CAPI_DGLSaveGraphs")
.set_body([](DGLArgs args, DGLRetValue *rv) {
    void *graph_list = args[0];
    int graph_list_size = args[1];
    void *node_feat_name_list = args[2];
    void *node_feat_list = args[3];
    int node_feat_list_size = args[4];
    void *edge_feat_name_list = args[5];
    void *edge_feat_list = args[6];
    int edge_feat_list_size = args[7];
    std::string filename = args[8];


    GraphHandle *inhandles = static_cast<GraphHandle *>(graph_list);
    std::vector<const ImmutableGraph *> graphs;
    for (int i = 0; i < graph_list_size; ++ i) {
      const GraphInterface *ptr = static_cast<const GraphInterface *>(inhandles[i]);
      const ImmutableGraph* g = ToImmutableGraph(ptr);
      graphs.push_back(g);
    }

    std::vector<NDArray> node_feats;
    std::vector<NDArray> edge_feats;
    std::vector<std::string> node_names;
    std::vector<std::string> edge_names;
    ToNameAndTensorList(node_feat_list, node_feat_name_list, node_feat_list_size, node_names,
                        node_feats);
    ToNameAndTensorList(edge_feat_list, edge_feat_name_list, edge_feat_list_size, edge_names,
                        edge_feats);




});

const ImmutableGraph* ToImmutableGraph(const GraphInterface *g) {
  const ImmutableGraph *imgr = dynamic_cast<const ImmutableGraph *>(g);
  if (g) {
    return imgr;
  } else {
    const Graph *mgr = dynamic_cast<const Graph *>(g);
    CHECK(mgr) << "Invalid Graph Pointer";
    IdArray srcs_array = mgr->Edges("srcdst").src;
    IdArray dsts_array = mgr->Edges("srcdst").dst;
    COOPtr coo(new COO(mgr->NumVertices(), srcs_array, dsts_array, mgr->IsMultigraph()));
    const ImmutableGraph* imgptr =new ImmutableGraph(coo);
    return imgptr;
  }
}

bool SaveDGLGraphs(std::vector<ImmutableGraph*> graph_list,
                   std::vector<NDArray> node_feats,
                   std::vector<NDArray> edge_feats,
                   std::vector<std::string> node_names,
                   std::vector<std::string> edge_names,
                   NDArray label_list,
                   const std::string& filename) {


  SeekStream *fs = SeekStream::CreateForRead(filename.c_str());
  fs->Write(runtime::kDGLNDArrayMagic);
  fs->Write(kVersion);
  fs->Write(kImmutableGraph);

  fs->Seek(4096);
  dgl_id_t num_graph = graph_list.size();
  fs->Write(num_graph);
  size_t indices_start_ptr = fs->Tell();
  std::vector<uint64_t> graph_indices(num_graph);
  fs->Write(graph_indices);


  std::vector<dgl_id_t> nodes_num_list;
  std::vector<dgl_id_t> edges_num_list;
  for (int i = 0; i < num_graph; ++ i) {
    nodes_num_list.push_back(graph_list[i]->NumVertices());
    edges_num_list.push_back(graph_list[i]->NumEdges());
  }
  VecToIdArray(nodes_num_list).Save(fs);
  VecToIdArray(edges_num_list).Save(fs);

  for (int i = 0; i < num_graph; ++ i) {
    graph_indices[i] = fs->Tell();
    ImmutableGraph *g = graph_list[i];
    g->GetInCSR()->indptr().Save(fs);
    g->GetInCSR()->indices().Save(fs);
    g->GetInCSR()->edge_ids().Save(fs);
    fs->Write(node_feats.size());
    fs->Write(edge_feats.size());
    for (int j = 0; j < node_feats.size(); ++ j) {
      fs->Write(node_names[j]);
      node_feats[j].Save(fs);
    }
    for (int k = 0; k < edge_feats.size(); ++ k) {
      fs->Write(edge_names[k]);
      edge_feats[k].Save(fs);
    }
  }
  fs->Seek(indices_start_ptr);
  fs->Write(graph_indices);
  return true;
}


}
}

