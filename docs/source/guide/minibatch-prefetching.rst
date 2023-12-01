.. _guide-minibatch-prefetching:

6.8 Feature Prefetching
-----------------------

In minibatch training of GNNs, especially with neighbor sampling approaches, we
often see that a large amount of node features need to be copied to the device
for computing GNNs. To mitigate this bottleneck of data movement, GraphBolt
supports *feature prefetching* so that the model computation and data movement
can happen in parallel. Such parallelism is totally transparent to users and
requires no extra code.

Below is a simple example of how to enable feature prefetching in GraphBolt.

.. code:: python

    # Other data pipe code.
    datapipe = datapipe.fetch_feature(
        features,
        node_feature_keys=["feat"],
        edge_feature_keys=["feat"]
    )
    datapipe = datapipe.copy_to(device=device)

During data loading, GraphBolt will prefetch the features specified in
``node_feature_keys`` and ``edge_feature_keys`` and save them to the
``node_features`` and ``edge_features`` of the generated minibatch.
