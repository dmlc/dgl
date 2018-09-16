#include <dgl/runtime/packed_func.h>
#include <dgl/runtime/registry.h>
#include <dgl/graph.h>

using tvm::runtime::TVMArgs;
using tvm::runtime::TVMArgValue;
using tvm::runtime::TVMRetValue;
using tvm::runtime::PackedFunc;

namespace dgl {

typedef void* GraphHandle;

void DGLGraphCreate(TVMArgs args, TVMRetValue* rv) {
  GraphHandle ghandle = new Graph();
  *rv = ghandle;
}

TVM_REGISTER_GLOBAL("cgraph._CAPI_DGLGraphCreate")
.set_body(DGLGraphCreate);

void DGLGraphFree(TVMArgs args, TVMRetValue* rv) {
  GraphHandle ghandle = args[0];
  Graph* gptr = static_cast<Graph*>(ghandle);
  delete gptr;
}

TVM_REGISTER_GLOBAL("cgraph._CAPI_DGLGraphFree")
.set_body(DGLGraphFree);

void DGLGraphAddVertices(TVMArgs args, TVMRetValue* rv) {
  GraphHandle ghandle = args[0];
  Graph* gptr = static_cast<Graph*>(ghandle);
  uint64_t num_vertices = args[1];
  gptr->AddVertices(num_vertices);
}

TVM_REGISTER_GLOBAL("cgraph._CAPI_DGLGraphAddVertices")
.set_body(DGLGraphAddVertices);

void DGLGraphAddEdge(TVMArgs args, TVMRetValue* rv) {
  GraphHandle ghandle = args[0];
  Graph* gptr = static_cast<Graph*>(ghandle);
  const dgl_id_t src = args[1];
  const dgl_id_t dst = args[2];
  gptr->AddEdge(src, dst);
}

TVM_REGISTER_GLOBAL("cgraph._CAPI_DGLGraphAddEdge")
.set_body(DGLGraphAddEdge);

void DGLGraphAddEdges(TVMArgs args, TVMRetValue* rv) {
  GraphHandle ghandle = args[0];
  Graph* gptr = static_cast<Graph*>(ghandle);
  const IdArray src = args[1];
  const IdArray dst = args[2];
  gptr->AddEdges(src, dst);
}

TVM_REGISTER_GLOBAL("cgraph._CAPI_DGLGraphAddEdges")
.set_body(DGLGraphAddEdges);

void DGLGraphNumVertices(TVMArgs args, TVMRetValue* rv) {
  GraphHandle ghandle = args[0];
  const Graph* gptr = static_cast<Graph*>(ghandle);
  *rv = static_cast<int64_t>(gptr->NumVertices());
}

TVM_REGISTER_GLOBAL("cgraph._CAPI_DGLGraphNumVertices")
.set_body(DGLGraphNumVertices);

void DGLGraphNumEdges(TVMArgs args, TVMRetValue* rv) {
  GraphHandle ghandle = args[0];
  const Graph* gptr = static_cast<Graph*>(ghandle);
  *rv = static_cast<int64_t>(gptr->NumEdges());
}

TVM_REGISTER_GLOBAL("cgraph._CAPI_DGLGraphNumEdges")
.set_body(DGLGraphNumEdges);

}  // namespace dgl
