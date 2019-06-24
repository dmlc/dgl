//
// Created by Allen Zhou on 2019/6/19.
//

#include <tuple>
#include "graph_serialize.h"
#include <dmlc/io.h>
#include <dmlc/type_traits.h>

using dgl::COO;
using dgl::COOPtr;
using dgl::ImmutableGraph;
using dmlc::SeekStream;
//using dmlc::io::LocalFileSystem;
using dgl::runtime::NDArray;

namespace dmlc { DMLC_DECLARE_TRAITS(has_saveload, NDArray, true); }

namespace dgl {
namespace serialize {

typedef std::pair<std::string, NDArray> NamedTensor;

enum GraphType {
  kMutableGraph = 0,
  kImmutableGraph = 1
};


struct ImmutableGraphSerialize {
  CSRPtr csrPtr;
  std::vector<NamedTensor> node_feats;
  std::vector<NamedTensor> edge_feats;
};

struct GraphBinary {

  struct HeadData {
    int DGLMagicNumber;
    int kVersion;
    GraphType graphType;
  };

  struct MetaData {
    uint32_t num_graphs;
    std::vector<uint64_t> g_indices;
    std::vector<NamedTensor> label_list;
    std::vector<dgl_id_t> node_num_list;
    std::vector<dgl_id_t> edge_num_list;
  };

  std::vector<ImmutableGraphSerialize> g_list;

};

struct DGLGraphSerialize {
  void *g_handle;
  uint32_t num_node_feats;
  uint32_t num_edge_feats;
  wchar_t **node_names;
  NDArray *node_feats;
  wchar_t **edge_names;
  NDArray *edge_feats;

};

DGL_REGISTER_GLOBAL("graph_serialize._CAPI_DGLSaveGraphsV2")
.set_body([](DGLArgs args, DGLRetValue *rv) {
    LOG(INFO) << "Good";

    void *g_list_handle = args[0];
    LOG(INFO) << "Good2";
    int num_graph = args[1];
    std::string filename = args[2];
    const DGLGraphSerialize* g_list= static_cast<const DGLGraphSerialize*>(g_list_handle);
//    const char *name_handle = static_cast<const char *>(filename_handle);
//    std::string filename(name_handle);
    LOG(INFO) << filename;
    LOG(INFO) << g_list;
//    LoadDGLGraphs(filename, idx_list);
});


DGL_REGISTER_GLOBAL("graph_serialize._CAPI_DGLLoadGraphs")
.set_body([](DGLArgs args, DGLRetValue *rv) {
    void *filename_handle = args[0];
    void *idx_list_handle = args[1];
    int num_idx = args[2];
    uint32_t *idx_list_array = static_cast<uint32_t *>(idx_list_handle);
    std::vector<uint32_t> idx_list(idx_list_array, idx_list_array + num_idx);
    const char *name_handle = static_cast<const char *>(filename_handle);
    std::string filename(name_handle);
    LoadDGLGraphs(filename, idx_list);
});


DGL_REGISTER_GLOBAL("graph_serialize._CAPI_DGLSaveGraphs")
.set_body([](DGLArgs args, DGLRetValue *rv) {
    LOG(INFO) << "SaveGraph blablabla";
    void *graph_list = args[0];
    int graph_list_size = args[1];
    void *node_feat_name_list = args[2];
    void *node_feat_list = args[3];
    int node_feat_list_size = args[4];
    void *edge_feat_name_list = args[5];
    void *edge_feat_list = args[6];
    int edge_feat_list_size = args[7];
    std::string filename = args[8];
    LOG(INFO) << filename;

    GraphHandle *inhandles = static_cast<GraphHandle *>(graph_list);
    std::vector<const ImmutableGraph *> graphs;
    for (int i = 0; i < graph_list_size; ++i) {
      const GraphInterface *ptr = static_cast<const GraphInterface *>(inhandles[i]);
      const ImmutableGraph *g = ToImmutableGraph(ptr);
      graphs.push_back(g);
    }

    std::vector<NamedTensor> node_feats = ToNamedTensorList(node_feat_list, node_feat_name_list,
                                                            node_feat_list_size);
    std::vector<NamedTensor> edge_feats = ToNamedTensorList(edge_feat_list, edge_feat_name_list,
                                                            edge_feat_list_size);


//    SaveDGLGraphs(graphs, node_feats, edge_feats, filename);


});

std::vector<NamedTensor> ToNamedTensorList(void *pytensorlist, void *pynamelist, int list_size) {
  const NDArray *tensor_list_handle = static_cast<const NDArray *>(pytensorlist);
  const char **name_list_handle = static_cast<const char **>(pynamelist);
  std::vector<NamedTensor> named_tensor_list;
  for (int i = 0; i < list_size; i ++) {
    std::string name(name_list_handle[i]);
    named_tensor_list.emplace_back(name, tensor_list_handle[i]);
  }
  return named_tensor_list;
}

void ToNameAndTensorList(void *pytensorlist, void *pynamelist, int list_size,
                         std::vector<std::string> &name_listptr,
                         std::vector<NDArray> &tensor_listptr) {
  const NDArray *tensor_list_handle = static_cast<const NDArray *>(pytensorlist);
  const char **name_list_handle = static_cast<const char **>(pynamelist);
  for (int i = 0; i < list_size; i++) {
    tensor_listptr.push_back(tensor_list_handle[i]);
    name_listptr.emplace_back(name_list_handle[i]);
  }
}

const ImmutableGraph *ToImmutableGraph(const GraphInterface *g) {
  const ImmutableGraph *imgr = dynamic_cast<const ImmutableGraph *>(g);
  if (imgr) {
    return imgr;
  } else {
    const Graph *mgr = dynamic_cast<const Graph *>(g);
    CHECK(mgr) << "Invalid Graph Pointer";
    IdArray srcs_array = mgr->Edges("srcdst").src;
    IdArray dsts_array = mgr->Edges("srcdst").dst;
    COOPtr coo(new COO(mgr->NumVertices(), srcs_array, dsts_array, mgr->IsMultigraph()));
    const ImmutableGraph *imgptr = new ImmutableGraph(coo);
    return imgptr;
  }
}

bool SaveDGLGraphs(std::vector<const ImmutableGraphSerialize *> graph_list,
                   std::vector<NamedTensor> label_list,
                   const std::string &filename) {


  auto *fs = dynamic_cast<SeekStream *>(SeekStream::Create(filename.c_str(), "w",
                                                           true));
  CHECK(fs) << "File name is not a valid local file name";
  fs->Write(runtime::kDGLNDArrayMagic);
  const uint32_t kVersion = 1;
  fs->Write(kVersion);
  fs->Write(kImmutableGraph);

  fs->Seek(4096);
  dgl_id_t num_graph = graph_list.size();
  fs->Write(num_graph);
  size_t indices_start_ptr = fs->Tell();
  std::vector<uint64_t> graph_indices(num_graph);
  fs->Write(graph_indices);

  std::vector<dgl_id_t> nodes_num_list(num_graph);
  std::vector<dgl_id_t> edges_num_list(num_graph);
  for (int i = 0; i < num_graph; ++i) {
    const CSRPtr g_csr = graph_list[i]->csrPtr;
    nodes_num_list[i] = g_csr->NumVertices();
    edges_num_list[i] = g_csr->NumEdges();
  }
  fs->Write(nodes_num_list);
  fs->Write(edges_num_list);

  for (int i = 0; i < num_graph; ++i) {
    graph_indices[i] = fs->Tell();
    const ImmutableGraphSerialize *&g_struct = graph_list[i];
    const CSRPtr g_csr = g_struct->csrPtr;
    fs->Write(g_csr->indptr());
    fs->Write(g_csr->indices());
    fs->Write(g_csr->edge_ids());
    fs->Write(g_struct->node_feats);
    fs->Write(g_struct->edge_feats);
  }
  fs->Seek(indices_start_ptr);
  fs->Write(graph_indices);
  return true;
}

bool LoadDGLGraphs(const std::string &filename,
                   std::vector<uint32_t> idx_list) {
  SeekStream *fs = SeekStream::CreateForRead(filename.c_str(), true);
  uint64_t DGLMagicNum;
  uint64_t kImmutableGraph;
  CHECK_EQ(fs->Read(&DGLMagicNum), runtime::kDGLNDArrayMagic) << "Invalid DGL files";
  CHECK(fs->Read(&kImmutableGraph)) << "Invalid DGL files";
  uint32_t kVersion = 1;
  CHECK(fs->Read(&kVersion)) << "Invalid Serialization Version";
  fs->Seek(4096);
  dgl_id_t num_graph;
  CHECK(fs->Read(&num_graph)) << "Invalid number of graphs";
  std::vector<uint64_t> graph_indices(num_graph);
  CHECK(fs->ReadArray(&graph_indices[0], num_graph)) << "Invalid graph indices data";
  std::vector<dgl_id_t> nodes_num_list(num_graph);
  std::vector<dgl_id_t> edges_num_list(num_graph);
  CHECK(fs->ReadArray(&nodes_num_list, num_graph));
  CHECK(fs->ReadArray(&edges_num_list, num_graph));

  std::sort(idx_list.begin(), idx_list.end());
  std::vector<ImmutableGraphSerialize> graph_list;

  for (int i = 0; i < idx_list.size(); ++i) {
    fs->Seek(graph_indices[i]);
    NDArray indptr, indices, edge_id;
    indptr.Load(fs);
    indices.Load(fs);
    edge_id.Load(fs);
    CSRPtr csr = std::make_shared<CSR>(indptr, indices, edge_id);
    ImmutableGraph g(csr);
    size_t num_node_feats, num_edge_feats;
    fs->Read(&num_node_feats);
    fs->Read(&num_edge_feats);
    std::vector<NDArray> node_feats(num_node_feats);
    std::vector<NDArray> edge_feats(num_edge_feats);
    std::vector<std::string> node_names(num_node_feats);
    std::vector<std::string> edge_names(num_edge_feats);

    for (int j = 0; j < num_node_feats; ++j) {
      fs->Read(&node_names[j]);
      node_feats[j].Load(fs);
    }
    for (int k = 0; k < num_edge_feats; ++k) {
      fs->Read(&edge_names[k]);
      edge_feats[k].Load(fs);
    }


  }


}


}
}


//class Writer{
//public:
//  Writer(std::string filename){
//    this->current_pos=0;
//    this->fs= Stream::Create(filename.c_str(), "w", true);
//    CHECK(this->fs) << "Failed to initialize File Stream";
//  }
//
//  template <typename T>
//  void Write(T obj){
//    fs->Write(obj);
//    current_pos+= sizeof(obj);
//  }
//
//  void Write(NDArray ndarray){
////    ndarray->data
//    fs->Write(ndarray->ndim);
//    fs->Write(ndarray->dtype);
//    int ndim = ndarray->ndim;
//    fs->WriteArray(ndarray->shape, ndim);
//
//    int type_bytes = ndarray->dtype.bits / 8;
//    int64_t num_elems = 1;
//    for (int i = 0; i < ndim; ++i) {
//      num_elems *= ndarray->shape[i];
//    }
//    int64_t data_byte_size = type_bytes * num_elems;
//    fs->Write(data_byte_size);
//
//    if (DMLC_IO_NO_ENDIAN_SWAP &&
//        ndarray->ctx.device_type == kDLCPU &&
//        ndarray->strides == nullptr &&
//        ndarray->byte_offset == 0) {
//      // quick path
//      fs->Write(ndarray->data, data_byte_size);
//    }
//
//
//
//
//    ndarray.GetSize();
//    ndarray->data;
//    ndarray->byte_offset;
//    ndarray.Save(fs);
//    current_pos+=ndarray.GetSize();
//  }
//
//  uint64_t current_pos;
//
//  protected:
//    Stream* fs;
//};
