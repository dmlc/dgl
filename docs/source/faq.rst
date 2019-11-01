FAQ
===

Troubleshooting
----------------

Deep Graph Library (DGL) is still in its alpha stage, so expect some trial and error. Keep in mind that
DGL is a framework atop other frameworks, e.g., PyTorch, MXNet, so it is important
to figure out whether a bug is due to DGL or the backend framework. For example,
DGL will usually complain and throw a ``DGLError`` if anything goes wrong. If you
are pretty confident that it is a bug, feel free to raise an issue.


Out-of-memory
-------------

Graph can be very large and training on graph may cause out of memory (OOM) errors. There are several
tips to check when you get an OOM error.

* Try to avoid propagating node features to edges. Number of edges are usually
  much larger than number of nodes. Try to use out built-in functions whenever
  it is possible.
* Look out for cyclic references due to user-defined functions. Usually we recommend
  using global function or module class for the user-defined functions. Pay
  attention to the variables in function closure. Also, it is usually better to
  directly provide the UDFs in the message passing APIs rather than register them:

  ::

     # define a message function
     def mfunc(edges): return edges.data['x']

     # better as the graph `mfunc` does not hold a reference to `mfunc`
     g.send(some_edges, mfunc)

     # the graph hold a reference to `mfunc` so as all the variables in its closure
     g.register(mfunc)
     g.send(some_edges)

* If your scenario does not require autograd, you can use ``inplace=True`` flag
  in the message passing APIs. This will update features inplacely that might
  save memory.
