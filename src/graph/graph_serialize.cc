//
// Created by Allen Zhou on 2019/6/19.
//

#include <tuple>
#include "graph_serialize.h"
#include <dmlc/io.h>
#include <dmlc/type_traits.h>
#include <iostream>
#include <dgl/runtime/container.h>
#include <dgl/immutable_graph.h>
#include <dgl/runtime/object.h>

using namespace dgl::runtime;

using dgl::COO;
using dgl::COOPtr;
using dgl::ImmutableGraph;
//using dgl::ImGraphPtr;
using dmlc::SeekStream;
using dgl::runtime::NDArray;
using std::vector;
using dgl::serialize::GraphData;
namespace dmlc { DMLC_DECLARE_TRAITS(has_saveload, NDArray, true); }

namespace dgl {
namespace serialize {

enum GraphType {
  kMutableGraph = 0,
  kImmutableGraph = 1
};

//DGL_REGISTER_GLOBAL("graph_serialize._CAPI_MakeND")
//.set_body([](DGLArgs args, DGLRetValue *rv) {
//    ND nd = ND::Create();
//    NDArray nd_handle = args[0];
//    nd->tensor = &nd_handle;
//    *rv = nd;
//});



DGL_REGISTER_GLOBAL("graph_serialize._CAPI_MakeGraphData")
.set_body([](
DGLArgs args, DGLRetValue
*rv) {
GraphRef gptr = args[0];
ImGraphPtr imGPtr = std::dynamic_pointer_cast<ImmutableGraph>(gptr.sptr());
//    List<>
Map <std::string, Value> node_tensors = args[1];
Map <std::string, Value> edge_tensors = args[2];
GraphData gd = GraphData::Create();
gd->
setData(imGPtr, node_tensors, edge_tensors
);
*
rv = gd;
}
);

DGL_REGISTER_GLOBAL("graph_serialize._CAPI_DGLSaveGraphs")
.set_body([](DGLArgs args, DGLRetValue *rv) {
    std::string filename = args[0];
List <GraphData> graph_data = args[1];
SaveDGLGraphs(filename,
        graph_data
);
}
);

//
bool SaveDGLGraphs(std::string filename,
                   List <GraphData> graph_data) {
  auto *fs = dynamic_cast<SeekStream *>(SeekStream::Create(filename.c_str(), "w",
                                                           true));
  CHECK(fs) << "File name is not a valid local file name";
  fs->Write(runtime::kDGLNDArrayMagic);
  const uint32_t kVersion = 1;
  fs->Write(kVersion);
  fs->Write(kImmutableGraph);

  fs->Seek(4096);
  size_t num_graph = graph_data.size();
  fs->Write(num_graph);
  size_t indices_start_ptr = fs->Tell();
  std::vector<uint64_t> graph_indices(num_graph);
  fs->Write(graph_indices);

  std::vector<dgl_id_t> nodes_num_list(num_graph);
  std::vector<dgl_id_t> edges_num_list(num_graph);


  for (int i = 0; i < num_graph; ++ i) {
    nodes_num_list[i] = graph_data[i]->gptr->NumVertices();
    edges_num_list[i] = graph_data[i]->gptr->NumEdges();
  }
  fs->Write(nodes_num_list);
  fs->Write(edges_num_list);

  for (int i = 0; i < num_graph; ++ i) {
    graph_indices[i] = fs->Tell();
    const CSRPtr g_csr = graph_data[i]->gptr->GetInCSR();
    fs->Write(g_csr->indptr());
    fs->Write(g_csr->indices());
    fs->Write(g_csr->edge_ids());
    fs->Write(graph_data[i]->node_tensors.size());
    for (const auto &node_tensorkv: graph_data[i]->node_tensors) {
      std::string name = node_tensorkv.first;
      NDArray tensor = node_tensorkv.second;
      fs->Write(name);
      fs->Write(tensor);
    }
    for (const auto &edge_tensorskv: graph_data[i]->node_tensors) {
      std::string name = edge_tensorskv.first;
      NDArray tensor = edge_tensorskv.second;
      fs->Write(name);
      fs->Write(tensor);
    }
  }
  fs->Seek(indices_start_ptr);
  fs->Write(graph_indices);

  fs->Seek(indices_start_ptr);
  fs->Write(graph_indices);
  return true;
}

DGL_REGISTER_GLOBAL("graph_serialize._CAPI_DGLLoadGraphs")
.set_body([](DGLArgs args, DGLRetValue *rv) {
std::string filename = args[0];
List <Value> *idx_list_handle = args[1];
    int num_idx = args[2];
    uint32_t *idx_list_array = static_cast<uint32_t *>(idx_list_handle);
    std::vector<uint32_t> idx_list(idx_list_array, idx_list_array + num_idx);


});


//DGL_REGISTER_GLOBAL("graph_serialize._CAPI_DGLLoadGraphs")
//.set_body([](DGLArgs args, DGLRetValue *rv) {
//    void *filename_handle = args[0];
//    void *idx_list_handle = args[1];
//    int num_idx = args[2];
//    uint32_t *idx_list_array = static_cast<uint32_t *>(idx_list_handle);
//    std::vector<uint32_t> idx_list(idx_list_array, idx_list_array + num_idx);
//    const char *name_handle = static_cast<const char *>(filename_handle);
//    std::string filename(name_handle);
//    *rv = reinterpret_cast<void *>(&LoadDGLGraphs(filename, idx_list)[0]);
//});
//
//const ImmutableGraph *ToImmutableGraph(const GraphInterface *g) {
//  const ImmutableGraph *imgr = dynamic_cast<const ImmutableGraph *>(g);
//  if (imgr) {
//    return imgr;
//  } else {
//    const Graph *mgr = dynamic_cast<const Graph *>(g);
//    CHECK(mgr) << "Invalid Graph Pointer";
//    IdArray srcs_array = mgr->Edges("srcdst").src;
//    IdArray dsts_array = mgr->Edges("srcdst").dst;
//    // TODO: Memory Leak
//    COOPtr coo(new COO(mgr->NumVertices(), srcs_array, dsts_array, mgr->IsMultigraph()));
//    const ImmutableGraph *imgptr = new ImmutableGraph(coo);
//    return imgptr;
//  }
//}
//
//std::vector<DGLGraphSerialize> LoadDGLGraphs(const std::string &filename,
//                                             std::vector<uint32_t> idx_list) {
//  SeekStream *fs = SeekStream::CreateForRead(filename.c_str(), true);
//  uint64_t DGLMagicNum;
//  uint64_t kImmutableGraph;
//  CHECK_EQ(fs->Read(&DGLMagicNum), runtime::kDGLNDArrayMagic) << "Invalid DGL files";
//  CHECK(fs->Read(&kImmutableGraph)) << "Invalid DGL files";
//  uint32_t kVersion = 1;
//  CHECK(fs->Read(&kVersion)) << "Invalid Serialization Version";
//  fs->Seek(4096);
//  dgl_id_t num_graph;
//  CHECK(fs->Read(&num_graph)) << "Invalid number of graphs";
//  std::vector<uint64_t> graph_indices(num_graph);
//  CHECK(fs->ReadArray(&graph_indices[0], num_graph)) << "Invalid graph indices data";
//  std::vector<dgl_id_t> nodes_num_list(num_graph);
//  std::vector<dgl_id_t> edges_num_list(num_graph);
//  CHECK(fs->ReadArray(&nodes_num_list, num_graph));
//  CHECK(fs->ReadArray(&edges_num_list, num_graph));
//
//  std::sort(idx_list.begin(), idx_list.end());
////  std::vector<DGLGraphSerialize> graph_list;
//
//
//  for (int i = 0; i < idx_list.size(); ++ i) {
//    fs->Seek(graph_indices[i]);
//    NDArray indptr, indices, edge_id;
//    uint32_t num_node_feats, num_edge_feats;
//    const char **node_names, **edge_names;
//    NDArray *node_feats, *edge_feats;
//    fs->Read(&indptr);
//    fs->Read(&indices);
//    fs->Read(&edge_id);
//    fs->Read(&num_node_feats);
//    fs->ReadArray(&node_names, num_node_feats);
//    fs->ReadArray(&node_feats, num_node_feats);
//    fs->Read(&num_edge_feats);
//    fs->ReadArray(&edge_names, num_edge_feats);
//    fs->ReadArray(&edge_feats, num_edge_feats);
//    CSRPtr csr = std::make_shared<CSR>(indptr, indices, edge_id);
//    ImmutableGraph g(csr);
////    ret_g.emplace_back(csr);
//    DGLGraphSerialize g_struct = {.g_handle= g.Reset(),
//            .num_node_feats = num_node_feats,
//            .num_edge_feats = num_node_feats,
//            .node_names = node_names,
//            .node_feats = nullptr,
//            .edge_names = edge_names,
//            .edge_feats = nullptr};
////    DGLGraphSerialize g_struct = {.g_handle= g.Reset(),
////            .num_node_feats = num_node_feats,
////            .num_edge_feats = num_node_feats,
////            .node_names = node_names,
////            .node_feats = &node_feats,
////            .edge_names = edge_names,
////            .edge_feats = &edge_feats};
//    graph_list.push_back(g_struct);
//  }
//  return graph_list;
//}
}

}


