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

Reproducibility
---------------
Like PyTorch, we also do not guarantee completely reproducible results across multiple releases,
individual commits or different platforms.

However, we guarantee determinism on both CPU and GPU for most of the operators defined in ``dgl.ops`` (and
thus built-in message-passing functions) from DGL v0.5 on, this being said you will get exactly the same
output/gradients in multiple runs by fixing the random seed of Python, Numpy, and backend framework. You are
expected to get the same training loss/accuracy if your program only uses deterministic operators in backend
framework (for PyTorch, see https://pytorch.org/docs/stable/notes/randomness.html) and deterministic DGL
message-passing operators/functions.

For message-passing, we do not guarantee the determinism only in following cases:

1. The backward phase of Min/Max reduce function (we depend on ``scatter_add_`` operator in backend frameworks,
   and it's not guaranteed to be deterministic).
2. Message Passing on ``DGLGraph``'s with restricted format ``COO`` (this will only happen when user specifies
   ``formats='coo'`` when creating the graph, normal users should not specify ``formats`` argument, which is
   only designed for expert users to handle extremely large graph).

Note that though operators above are not deterministic, the difference across multiple runs is quite small.
