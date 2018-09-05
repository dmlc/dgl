#include <dgl/runtime/packed_func.h>
#include <dgl/runtime/registry.h>

using namespace tvm;
using namespace tvm::runtime;

void MyAdd(TVMArgs args, TVMRetValue* rv) {
  int a = args[0];
  int b = args[1];
  *rv = a + b;
}

void CallPacked() {
  PackedFunc myadd = PackedFunc(MyAdd);
  int c = myadd(1, 2);
}

TVM_REGISTER_GLOBAL("myadd")
.set_body(MyAdd);
