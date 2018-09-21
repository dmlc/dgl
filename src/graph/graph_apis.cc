#include <dgl/runtime/packed_func.h>
#include <dgl/runtime/registry.h>
#include <dgl/graph.h>

using tvm::runtime::TVMArgs;
using tvm::runtime::TVMArgValue;
using tvm::runtime::TVMRetValue;
using tvm::runtime::PackedFunc;

namespace dgl {
namespace {
/*!\brief Convert EdgeArray structure to PackedFunc */
PackedFunc ConvertEdgeArrayToPackedFunc(const Graph::EdgeArray& ea) {
  auto body = [ea] (TVMArgs args, TVMRetValue* rv) {
      int which = args[0];
      if (which == 0) {
        *rv = ea.src;
      } else if (which == 1) {
        *rv = ea.dst;
      } else if (which == 2) {
        *rv = ea.id;
      } else {
        LOG(FATAL) << "invalid choice";
      }
    };
  return PackedFunc(body);
}

DLManagedTensor* CreateTmpDLManagedTensor(const TVMArgValue& arg) {
  const DLTensor* dl_tensor = arg;
  DLManagedTensor* ret = new DLManagedTensor();
  ret->deleter = [] (DLManagedTensor* self) { delete self; };
  ret->manager_ctx = nullptr;
  ret->dl_tensor = *dl_tensor;
  return ret;
}

}  // namespace

// Graph handler type
typedef void* GraphHandle;

TVM_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphCreate")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = new Graph();
    *rv = ghandle;
  });

TVM_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphFree")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = args[0];
    Graph* gptr = static_cast<Graph*>(ghandle);
    delete gptr;
  });

TVM_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphAddVertices")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = args[0];
    Graph* gptr = static_cast<Graph*>(ghandle);
    uint64_t num_vertices = args[1];
    gptr->AddVertices(num_vertices);
  });

TVM_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphAddEdge")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = args[0];
    Graph* gptr = static_cast<Graph*>(ghandle);
    const dgl_id_t src = args[1];
    const dgl_id_t dst = args[2];
    gptr->AddEdge(src, dst);
  });

TVM_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphAddEdges")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = args[0];
    Graph* gptr = static_cast<Graph*>(ghandle);
    const IdArray src = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[1]));
    const IdArray dst = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[2]));
    gptr->AddEdges(src, dst);
  });

TVM_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphClear")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = args[0];
    Graph* gptr = static_cast<Graph*>(ghandle);
    gptr->Clear();
  });

TVM_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphNumVertices")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = args[0];
    const Graph* gptr = static_cast<Graph*>(ghandle);
    *rv = static_cast<int64_t>(gptr->NumVertices());
  });

TVM_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphNumEdges")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = args[0];
    const Graph* gptr = static_cast<Graph*>(ghandle);
    *rv = static_cast<int64_t>(gptr->NumEdges());
  });

TVM_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphHasVertex")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = args[0];
    const Graph* gptr = static_cast<Graph*>(ghandle);
    const dgl_id_t vid = args[1];
    *rv = gptr->HasVertex(vid);
  });

TVM_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphHasVertices")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = args[0];
    const Graph* gptr = static_cast<Graph*>(ghandle);
    const IdArray vids = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[1]));
    *rv = gptr->HasVertices(vids);
  });

TVM_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphHasEdge")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = args[0];
    const Graph* gptr = static_cast<Graph*>(ghandle);
    const dgl_id_t src = args[1];
    const dgl_id_t dst = args[2];
    *rv = gptr->HasEdge(src, dst);
  });

TVM_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphHasEdges")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = args[0];
    const Graph* gptr = static_cast<Graph*>(ghandle);
    const IdArray src = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[1]));
    const IdArray dst = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[2]));
    *rv = gptr->HasEdges(src, dst);
  });

TVM_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphPredecessors")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = args[0];
    const Graph* gptr = static_cast<Graph*>(ghandle);
    const dgl_id_t vid = args[1];
    const uint64_t radius = args[2];
    *rv = gptr->Predecessors(vid, radius);
  });

TVM_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphSuccessors")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = args[0];
    const Graph* gptr = static_cast<Graph*>(ghandle);
    const dgl_id_t vid = args[1];
    const uint64_t radius = args[2];
    *rv = gptr->Successors(vid, radius);
  });

TVM_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphEdgeId")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = args[0];
    const Graph* gptr = static_cast<Graph*>(ghandle);
    const dgl_id_t src = args[1];
    const dgl_id_t dst = args[2];
    *rv = static_cast<int64_t>(gptr->EdgeId(src, dst));
  });

TVM_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphEdgeIds")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = args[0];
    const Graph* gptr = static_cast<Graph*>(ghandle);
    const IdArray src = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[1]));
    const IdArray dst = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[2]));
    *rv = gptr->EdgeIds(src, dst);
  });

TVM_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphInEdges_1")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = args[0];
    const Graph* gptr = static_cast<Graph*>(ghandle);
    const dgl_id_t vid = args[1];
    *rv = ConvertEdgeArrayToPackedFunc(gptr->InEdges(vid));
  });

TVM_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphInEdges_2")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = args[0];
    const Graph* gptr = static_cast<Graph*>(ghandle);
    const IdArray vids = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[1]));
    *rv = ConvertEdgeArrayToPackedFunc(gptr->InEdges(vids));
  });

TVM_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphOutEdges_1")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = args[0];
    const Graph* gptr = static_cast<Graph*>(ghandle);
    const dgl_id_t vid = args[1];
    *rv = ConvertEdgeArrayToPackedFunc(gptr->OutEdges(vid));
  });

TVM_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphOutEdges_2")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = args[0];
    const Graph* gptr = static_cast<Graph*>(ghandle);
    const IdArray vids = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[1]));
    *rv = ConvertEdgeArrayToPackedFunc(gptr->OutEdges(vids));
  });

TVM_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphEdges")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = args[0];
    const Graph* gptr = static_cast<Graph*>(ghandle);
    const bool sorted = args[1];
    *rv = ConvertEdgeArrayToPackedFunc(gptr->Edges(sorted));
  });

TVM_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphInDegree")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = args[0];
    const Graph* gptr = static_cast<Graph*>(ghandle);
    const dgl_id_t vid = args[1];
    *rv = static_cast<int64_t>(gptr->InDegree(vid));
  });

TVM_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphInDegrees")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = args[0];
    const Graph* gptr = static_cast<Graph*>(ghandle);
    const IdArray vids = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[1]));
    *rv = gptr->InDegrees(vids);
  });

TVM_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphOutDegree")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = args[0];
    const Graph* gptr = static_cast<Graph*>(ghandle);
    const dgl_id_t vid = args[1];
    *rv = static_cast<int64_t>(gptr->OutDegree(vid));
  });

TVM_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphOutDegrees")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = args[0];
    const Graph* gptr = static_cast<Graph*>(ghandle);
    const IdArray vids = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[1]));
    *rv = gptr->OutDegrees(vids);
  });

TVM_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphMerge")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    void* list = args[0];
    GraphHandle* inhandles = static_cast<GraphHandle*>(list);
    int list_size = args[1];
    std::vector<const Graph*> graphs;
    for (int i = 0; i < list_size; ++i) {
      const Graph* gr = static_cast<const Graph*>(inhandles[i]);
      graphs.push_back(gr);
    }
    Graph* gptr = new Graph();
    *gptr = Graph::Merge(std::move(graphs));
    GraphHandle ghandle = gptr;
    *rv = ghandle;
  });

}  // namespace dgl
