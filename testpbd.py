import dgl
from dgl._ffi._cy3.core import _return_pbd_object 

import sys
sys.path.append("./build/")

import dglpybind

# def main():
g=dgl.DGLGraph()
# ad = dglpybind.HasEdgeBetween(g._graph.handle, 0, 1)
# # ad2= dglpybind.HasEdgeBetween2(g._graph.handle, 0, 1)
# print("aaa {}".format(ad))
# # print("aaa2 {}".format(ad2))
# print(_return_pbd_object(ad))

new_g=dglpybind.new_gindex()

# add = int(input())
print(new_g)
print(_return_pbd_object(new_g))



    # input()
# s
# main()

# if __name__ == "__main__":
#     # main()  # Normal invocation; commented out, because we will trace it.

#     # The following (a) imports minimum dependencies, (b) ensures that
#     # output is immediately flushed (e.g. for segfaults), and (c) traces
#     # execution of your function, but filtering out any Python code outside
#     # of the system prefix.
#     import sys
#     import trace
#     sys.stdout = sys.stderr
#     tracer = trace.Trace(trace=1, count=0, ignoredirs=["/usr", sys.prefix])
#     tracer.runfunc(main)
