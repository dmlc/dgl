//
// Created by Allen Zhou on 2019/6/19.
//

#include "graph_seriazlie.h"

using dgl::COO;
using dgl::COOPtr;
using dgl::ImmutableGraph;
using dmlc::SeekStream;
using dgl::runtime::NDArray;

namespace dgl {
namespace serialize {


DGL_REGISTER_GLOBAL("serialize._CAPI_DGLSaveGraphs")
.set_body([](DGLArgs args, DGLRetValue *rv) {
    void *graph_list = args[0];
    int graph_list_size = args[1];
    void *node_feat_name_list = args[2];
    int node_feat_name_list_size = args[3];
    void *node_feat_list = args[4];
    int node_feat_list_size = args[5];
    void *edge_feat_name_list = args[6];
    int edge_feat_name_list_size = args[7];
    void *edge_feat_list = args[8];
    int edge_feat_list_size = args[9];
    std::string filename = args[10];


    GraphHandle *inhandles = static_cast<GraphHandle *>(graph_list);
    std::vector<const ImmutableGraph *> graphs;
    for (int i = 0; i < graph_list_size; ++ i) {
      const GraphInterface *ptr = static_cast<const GraphInterface *>(inhandles[i]);
      const ImmutableGraph* g = ToImmutableGraph(ptr);
      graphs.push_back(g);
    }

    Feat_Dict node_feats;
    Feat_Dict edge_feats;


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
        Feat_Dict node_feats,
        Feat_Dict edge_feats,
        std::string filename) {


  SeekStream *fs = SeekStream::CreateForRead(filename.c_str());
  fs->Write(runtime::kDGLNDArrayMagic);
  fs->Write(node_feats);
  fs->Write(edge_feats);


  return true;
}


}
}

