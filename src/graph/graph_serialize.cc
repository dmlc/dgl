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
//using namespace dmlc;

using dgl::COO;
using dgl::COOPtr;
using dgl::ImmutableGraph;
//using dgl::ImGraphPtr;
using dmlc::SeekStream;
using dgl::runtime::NDArray;
using std::vector;
using dgl::serialize::GraphData;
using dgl::serialize::GraphDataObject;

namespace dmlc {
DMLC_DECLARE_TRAITS(has_saveload, NDArray, true);
DMLC_DECLARE_TRAITS(has_saveload, GraphDataObject, true);
}

namespace dgl {
namespace serialize {

enum GraphType {
  kMutableGraph = 0ull,
  kImmutableGraph = 1ull
};

//DGL_REGISTER_GLOBAL("graph_serialize._CAPI_MakeND")
//.set_body([](DGLArgs args, DGLRetValue *rv) {
//    ND nd = ND::Create();
//    NDArray nd_handle = args[0];
//    nd->tensor = &nd_handle;
//    *rv = nd;
//});



DGL_REGISTER_GLOBAL("graph_serialize._CAPI_MakeGraphData")
.set_body([](DGLArgs args, DGLRetValue *rv) {
    GraphRef gptr = args[0];
    ImGraphPtr imGPtr = std::dynamic_pointer_cast<ImmutableGraph>(gptr.sptr());
//    List<>
    Map<std::string, Value> node_tensors = args[1];
    Map<std::string, Value> edge_tensors = args[2];
    GraphData gd = GraphData::Create();
    gd->setData(imGPtr, node_tensors, edge_tensors);
    *rv = gd;
});

DGL_REGISTER_GLOBAL("graph_serialize._CAPI_DGLSaveGraphs")
.set_body([](DGLArgs args, DGLRetValue *rv) {
    std::string filename = args[0];
    List<GraphData> graph_data = args[1];
    SaveDGLGraphs(filename, graph_data);
});

DGL_REGISTER_GLOBAL("graph_serialize._CAPI_GDataGraphHandle")
.set_body([](DGLArgs args, DGLRetValue *rv) {
    GraphData gdata = args[0];
    *rv = gdata -> gptr;
});

DGL_REGISTER_GLOBAL("graph_serialize._CAPI_GDataNodeTensors")
.set_body([](DGLArgs args, DGLRetValue *rv){
    GraphData gdata = args[0];
    Map<std::string, Value> rvmap;
    for (const auto& kv : gdata->node_tensors) {
      std::string k = kv.first;
      NDArray v = kv.second;
      Value vv = Value(MakeValue(v));
      rvmap.Set(k, vv);
    }
    // *rv = gdata->node_tensors[0].second;
    *rv = rvmap;
});

// DGL_REGISTER_GLOBAL("graph_serialize._CAPI_GDataNodeTensorsName")
// .set_body([](DGLArgs args, DGLRetValue *rv) {
//     GraphData gdata = args[0];
//     std::vector<NamedTensor> node_tensors = gdata->node_tensors;
//     std::vector<Value> name_list(node_tensors.size());
//     for (int i = 0; i < node_tensors.size(); ++i){
//       name_list[i]=MakeValue(node_tensors[i].first);
//     }
//     Map a();
//     *rv = List<Value>(name_list);
//     // Map<>
// });

constexpr uint64_t kDGLSerializeMagic = 0xDD2E4FF046B4A13F;

bool SaveDGLGraphs(std::string filename,
                   List<GraphData> graph_data) {
  auto *fs = dynamic_cast<SeekStream *>(SeekStream::Create(filename.c_str(), "w",
                                                           true));
  CHECK(fs) << "File name is not a valid local file name";

  // Write DGL MetaData
  const uint64_t kVersion = 1;
  fs->Write(kDGLSerializeMagic);
  fs->Write(kVersion);
  fs->Write(kImmutableGraph);
  fs->Seek(4096);

  // Write Graph Meta Data
  size_t num_graph = graph_data.size();

  std::vector<dgl_id_t> graph_indices(num_graph);
  std::vector<dgl_id_t> nodes_num_list(num_graph);
  std::vector<dgl_id_t> edges_num_list(num_graph);

  for (uint64_t i = 0; i < num_graph; ++ i) {
    nodes_num_list[i] = graph_data[i]->gptr->NumVertices();
    edges_num_list[i] = graph_data[i]->gptr->NumEdges();
  }
  // Reserve spaces for graph indices
  fs->Write(num_graph);
  size_t indices_start_ptr = fs->Tell();
  fs->Write(graph_indices);
  fs->Write(nodes_num_list);
  fs->Write(edges_num_list);

  // Write GraphData
  for (uint64_t i = 0; i < num_graph; ++ i) {
    graph_indices[i] = fs->Tell();
    GraphDataObject gdata= *graph_data[i].as<GraphDataObject>();
    fs->Write(gdata);
  }

  fs->Seek(indices_start_ptr);
  fs->Write(graph_indices);

  std::vector<dgl_id_t> test;
  fs->Seek(indices_start_ptr);
  fs->Read(&test);
  
  LOG(INFO) << graph_indices[0];
  LOG(INFO) << graph_indices[1];
  return true;
}


DGL_REGISTER_GLOBAL("graph_serialize._CAPI_DGLLoadGraphs")
.set_body([](DGLArgs args, DGLRetValue *rv) {
    std::string filename = args[0];
    List<Value> idxs = args[1];
    std::vector<size_t> idx_list(idxs.size());
    for (uint64_t i = 0; i < idxs.size(); ++ i) {
      idx_list[i] = static_cast<dgl_id_t >(idxs[i]->data);
    }
    *rv = List<GraphData>(LoadDGLGraphs(filename, idx_list));
});

std::vector<GraphData> LoadDGLGraphs(const std::string &filename,
                                     std::vector<dgl_id_t> idx_list) {
  SeekStream *fs = SeekStream::CreateForRead(filename.c_str(), true);
  // Read DGL MetaData
  uint64_t magicNum, graphType, version;
  fs->Read(&magicNum);
  fs->Read(&graphType);
  fs->Read(&version);
  fs->Seek(4096);

  CHECK_EQ(magicNum, kDGLSerializeMagic) << "Invalid DGL files";
  CHECK_EQ(graphType, GraphType::kImmutableGraph) << "Invalid DGL files";
  CHECK_EQ(version, 1) << "Invalid Serialization Version";

  // Read Graph MetaData
  uint64_t num_graph;
  CHECK(fs->Read(&num_graph)) << "Invalid num of graph";
  std::vector<dgl_id_t> graph_indices;
  std::vector<dgl_id_t> nodes_num_list;
  std::vector<dgl_id_t> edges_num_list;

  CHECK(fs->Read(&graph_indices)) << "Invalid graph indices";
  CHECK(fs->Read(&nodes_num_list)) << "Invalid node num list";
  CHECK(fs->Read(&edges_num_list)) << "Invalid edge num list";

  std::sort(idx_list.begin(), idx_list.end());

  // std::vector<std::shared_ptr<GraphDataObject>> graph_datas(idx_list.size());
  // for (int i = 0; i < idx_list.size(); ++ i) {
  //   fs->Seek(graph_indices[i]);
  //   graph_datas[i] = std::make_shared<GraphDataObject>();
  //   CHECK(fs->Read(graph_datas[i].get())) << "Invalid Graph";
  // }

  // std::vector<GraphData> gdata_refs;
  // for (int i = 0; i< graph_datas.size(); ++i){
  //   gdata_refs.emplace_back(graph_datas[i]);
  // }

  std::vector<GraphData> gdata_refs(idx_list.size());
  for (uint64_t i = 0; i< idx_list.size(); ++i){
    fs->Seek(graph_indices[i]);
    gdata_refs[i]=GraphData::Create();
    // TODO(jinjing): Any better solution avoid const_cast?
    GraphDataObject * gdata_ptr= const_cast<GraphDataObject *>(gdata_refs[i].as<GraphDataObject>());
    fs->Read(gdata_ptr);
  }

  return gdata_refs;
};


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
}

}


