.. _guide-minibatch-parallelism:

6.9 Data Loading Parallelism
-----------------------

In minibatch training of GNNs, we usually need to cover several stages to
generate a minibatch, including:

* Iterate over item set and generate minibatch seeds in batch size.
* Sample negative items for each seed from graph.
* Sample neighbors for each seed from graph.
* Exclude seed edges from the sampled subgraphs.
* Fetch node and edge features for the sampled subgraphs.
* Copy the MiniBatches to the target device.

.. code:: python

    datapipe = gb.ItemSampler(itemset, batch_size=1024, shuffle=True)
    datapipe = datapipe.sample_uniform_negative(g, 5)
    datapipe = datapipe.sample_neighbor(g, [10, 10]) # 2 layers.
    datapipe = datapipe.transform(gb.exclude_seed_edges)
    datapipe = datapipe.fetch_feature(feature, node_feature_keys=["feat"])
    datapipe = datapipe.copy_to(device)
    dataloader = gb.DataLoader(datapipe)

All these stages are implemented in separate
`IterableDataPipe <https://pytorch.org/data/0.7/torchdata.datapipes.iter.html>`__
and stacked together with `PyTorch DataLoader <https://pytorch.org/docs/stable/data
.html#torch.utils.data.DataLoader>`__.
This design allows us to easily customize the data loading process by
chaining different data pipes together. For example, if we want to sample
negative items for each seed from graph, we can simply chain the
:class:`~dgl.graphbolt.NegativeSampler` after the :class:`~dgl.graphbolt.ItemSampler`.

But simply chaining data pipes together incurs performance overheads as various
hardware resources such as CPU, GPU, PCIe, etc. are utilized by different stages.
As a result, the data loading mechanism is optimized to minimize the overheads
and achieve the best performance.

In specific, GraphBolt wraps the data pipes before ``fetch_feature`` with
multiprocessing which enables multiple processes to run in parallel. As for
``fetch_feature`` data pipe, we keep it running in the main process to avoid
data movement overheads between processes.

What's more, in order to overlap the data movement and model computation, we
wrap data pipes before ``copy_to`` with
`torchdata.datapipes.iter.Perfetcher <https://pytorch.org/data/0.7/generated/
torchdata.datapipes.iter.Prefetcher.html>`__
which prefetches elements from previous data pipes and puts them into a buffer.
Such prefetching is totally transparent to users and requires no extra code. It
brings a significant performance boost to minibatch training of GNNs.

Please refer to the source code of :class:`~dgl.graphbolt.DataLoader`
for more details.
