.. code:: ipython3

    # !pip install pytorch-lightning
    # !pip install torchlayers
    # !pip install torchmetric
    # !pip install dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html
    #!pip install rich --upgrade
    import pytorch_lightning as pl
    import torchmetrics
    import dgl
    import warnings
    warnings.filterwarnings('ignore')

.. code:: ipython3

    """
    Node Classification with DGL
    ============================
    GNNs are powerful tools for many machine learning tasks on graphs. In
    this introductory tutorial, you will learn the basic workflow of using
    GNNs for node classification, i.e. predicting the category of a node in
    a graph.
    By completing this tutorial, you will be able to
    -  Load a DGL-provided dataset.
    -  Build a GNN model with DGL-provided neural network modules.
    -  Train and evaluate a GNN model for node classification on either CPU
       or GPU.
    This tutorial assumes that you have experience in building neural
    networks with PyTorch.
    (Time estimate: 13 minutes)
    """
    
    import dgl
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

.. code:: ipython3

    ######################################################################
    # Overview of Node Classification with GNN
    # ----------------------------------------
    #
    # One of the most popular and widely adopted tasks on graph data is node
    # classification, where a model needs to predict the ground truth category
    # of each node. Before graph neural networks, many proposed methods are
    # using either connectivity alone (such as DeepWalk or node2vec), or simple
    # combinations of connectivity and the node's own features.  GNNs, by
    # contrast, offers an opportunity to obtain node representations by
    # combining the connectivity and features of a *local neighborhood*.
    #
    # `Kipf et
    # al., <https://arxiv.org/abs/1609.02907>`__ is an example that formulates
    # the node classification problem as a semi-supervised node classification
    # task. With the help of only a small portion of labeled nodes, a graph
    # neural network (GNN) can accurately predict the node category of the
    # others.
    # 
    # This tutorial will show how to build such a GNN for semi-supervised node
    # classification with only a small number of labels on the Cora
    # dataset,
    # a citation network with papers as nodes and citations as edges. The task
    # is to predict the category of a given paper. Each paper node contains a
    # word count vector as its features, normalized so that they sum up to one,
    # as described in Section 5.2 of
    # `the paper <https://arxiv.org/abs/1609.02907>`__.
    # 
    # Loading Cora Dataset
    # --------------------
    # 
    
    import dgl.data
    
    dataset = dgl.data.CoraGraphDataset()
    print('Number of categories:', dataset.num_classes)
    
    


.. parsed-literal::

      NumNodes: 2708
      NumEdges: 10556
      NumFeats: 1433
      NumClasses: 7
      NumTrainingSamples: 140
      NumValidationSamples: 500
      NumTestSamples: 1000
    Done loading data from cached files.
    Number of categories: 7
    

.. code:: ipython3

    
    ######################################################################
    # A DGL Dataset object may contain one or multiple graphs. The Cora
    # dataset used in this tutorial only consists of one single graph.
    # 
    
    g = dataset[0]
    
    
    ######################################################################
    # A DGL graph can store node features and edge features in two
    # dictionary-like attributes called ``ndata`` and ``edata``.
    # In the DGL Cora dataset, the graph contains the following node features:
    # 
    # - ``train_mask``: A boolean tensor indicating whether the node is in the
    #   training set.
    #
    # - ``val_mask``: A boolean tensor indicating whether the node is in the
    #   validation set.
    #
    # - ``test_mask``: A boolean tensor indicating whether the node is in the
    #   test set.
    #
    # - ``label``: The ground truth node category.
    #
    # -  ``feat``: The node features.
    # 
    print('Node features')
    print(g.ndata)
    print('Edge features')
    print(g.edata)


.. parsed-literal::

    Node features
    {'feat': tensor([[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
            ...,
            [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0526, 0.0000]]), 'label': tensor([4, 4, 4,  ..., 4, 3, 3]), 'test_mask': tensor([ True,  True, False,  ..., False, False, False]), 'train_mask': tensor([False, False, False,  ..., False, False, False]), 'val_mask': tensor([False, False,  True,  ..., False, False, False])}
    Edge features
    {'__orig__': tensor([  298,  9199,  1153,  ..., 10415,  5255,  6356])}
    

.. code:: ipython3

    ######################################################################
    # Defining a Graph Convolutional Network (GCN)
    # --------------------------------------------
    # 
    # This tutorial will build a two-layer `Graph Convolutional Network
    # (GCN) <http://tkipf.github.io/graph-convolutional-networks/>`__. Each
    # layer computes new node representations by aggregating neighbor
    # information.
    # 
    # To build a multi-layer GCN you can simply stack ``dgl.nn.GraphConv``
    # modules, which inherit ``torch.nn.Module``.
    # 

.. code:: ipython3

    from dgl.nn import GraphConv
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    
    class GCN(pl.LightningModule):
        def __init__(self, in_feats, h_feats, num_classes):
            super(GCN, self).__init__()
            self.conv1 = GraphConv(in_feats, h_feats)
            self.conv2 = GraphConv(h_feats, num_classes)
            self.train_accuracy = torchmetrics.Accuracy()
            self.val_accuracy = torchmetrics.Accuracy()
            self.test_accuracy = torchmetrics.Accuracy()
        
        def forward(self, g, in_feat):
            h = self.conv1(g, in_feat)
            h = F.relu(h)
            h = self.conv2(g, h)
            return h
        def training_step(self,batch,batch_idx):
          g = batch
          features = g.ndata['feat']
          labels = g.ndata['label']
          train_mask = g.ndata['train_mask']
          val_mask = g.ndata['val_mask']
          test_mask = g.ndata['test_mask']
          # Forward
          logits = self(g, features)
    
          # Compute prediction
          pred = logits.argmax(1)
    
          # Compute loss
          # Note that you should only compute the losses of the nodes in the training set.
          loss = F.cross_entropy(logits[train_mask], labels[train_mask])
    
          
          # Compute accuracy on training/validation/test
          self.train_accuracy(pred[train_mask],labels[train_mask])
          self.val_accuracy(pred[val_mask],labels[val_mask])
          self.test_accuracy(pred[test_mask],labels[test_mask])
          self.log_dict({'train/loss':loss,'train/accuracy':self.train_accuracy},prog_bar=True,on_step=True,on_epoch=True,batch_size=1) # set the batch_size =1
          return loss
    
        def validation_step(self,batch,batch_idx):
            self.log_dict({'valid/accuracy':self.val_accuracy,'test/accuracy':self.test_accuracy},prog_bar=True,on_step=True,on_epoch=True,batch_size=1) 
    
        def predict_step(self,batch,batch_idx):
          g = batch
          features = g.ndata['feat']
          # Forward
          logits = self(g, features)
          return logits
    
        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(),lr=1e-2,weight_decay=1e-7)
            lr_scheduler = {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.9,
                    patience=15,
                    min_lr=1e-3
                ),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "valid/accuracy_epoch",
                "strict": True,
                "name": "Learning Rate",
            }
            return [optimizer], [lr_scheduler]
    

.. code:: ipython3

    import os
    from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
    from pytorch_lightning.callbacks import RichProgressBar,ModelCheckpoint,EarlyStopping,LearningRateMonitor
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.getcwd(),
        filename=f'GCN_DGL_' + '{valid/accuracy:.6f}',
        monitor='valid/accuracy',
        mode='max',
        save_weights_only=False)
    
    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        version=1,
        name='GCN_by_DGL'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='valid/accuracy',
        min_delta=0.00,
        patience=15,
        verbose=True,
        mode='max',check_on_train_epoch_end=False, ##remember to disable check_on_train_epoch_end
    )
    
    lr_monitor = LearningRateMonitor(
        logging_interval='step'
    )
    
    trainer = pl.Trainer(
        fast_dev_run=False,
        max_epochs=100,
        gpus=1,
        precision=32,
        check_val_every_n_epoch=1,
        val_check_interval=1.0,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback, lr_monitor, early_stop_callback,RichProgressBar()],
        logger=logger
      )


.. parsed-literal::

    GPU available: True, used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    `Trainer(val_check_interval=1.0)` was configured so validation will run at the end of the training epoch..
    

.. code:: ipython3

    from torch.utils.data import DataLoader
    gloader = DataLoader([g],batch_size=None) # use "[g]" instead of "g" 
    model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes)
    loader_for_valid = DataLoader([0],batch_size=None) ## use an empty daloader to enable validation_step of lightning model

.. code:: ipython3

    trainer.fit(model, gloader , loader_for_valid)


.. parsed-literal::

    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    


.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┓
    ┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold">   </span>┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold"> Name           </span>┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold"> Type      </span>┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold"> Params </span>┃
    ┡━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━┩
    │<span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 0 </span>│ conv1          │ GraphConv │ 22.9 K │
    │<span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 1 </span>│ conv2          │ GraphConv │    119 │
    │<span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 2 </span>│ train_accuracy │ Accuracy  │      0 │
    │<span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 3 </span>│ val_accuracy   │ Accuracy  │      0 │
    │<span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 4 </span>│ test_accuracy  │ Accuracy  │      0 │
    └───┴────────────────┴───────────┴────────┘
    </pre>
    



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Trainable params</span>: 23.1 K                                                                     
    <span style="font-weight: bold">Non-trainable params</span>: 0                                                                      
    <span style="font-weight: bold">Total params</span>: 23.1 K                                                                         
    <span style="font-weight: bold">Total estimated model params size (MB)</span>: 0                                                    
    </pre>
    



.. parsed-literal::

    Output()


.. parsed-literal::

    Metric valid/accuracy improved. New best score: 0.116
    Metric valid/accuracy improved by 0.214 >= min_delta = 0.0. New best score: 0.330
    Metric valid/accuracy improved by 0.084 >= min_delta = 0.0. New best score: 0.414
    Metric valid/accuracy improved by 0.102 >= min_delta = 0.0. New best score: 0.516
    Metric valid/accuracy improved by 0.010 >= min_delta = 0.0. New best score: 0.526
    Metric valid/accuracy improved by 0.012 >= min_delta = 0.0. New best score: 0.538
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.540
    Metric valid/accuracy improved by 0.006 >= min_delta = 0.0. New best score: 0.546
    Metric valid/accuracy improved by 0.018 >= min_delta = 0.0. New best score: 0.564
    Metric valid/accuracy improved by 0.018 >= min_delta = 0.0. New best score: 0.582
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.584
    Metric valid/accuracy improved by 0.008 >= min_delta = 0.0. New best score: 0.592
    Metric valid/accuracy improved by 0.006 >= min_delta = 0.0. New best score: 0.598
    Metric valid/accuracy improved by 0.004 >= min_delta = 0.0. New best score: 0.602
    Metric valid/accuracy improved by 0.004 >= min_delta = 0.0. New best score: 0.606
    Metric valid/accuracy improved by 0.008 >= min_delta = 0.0. New best score: 0.614
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.616
    Metric valid/accuracy improved by 0.006 >= min_delta = 0.0. New best score: 0.622
    Metric valid/accuracy improved by 0.006 >= min_delta = 0.0. New best score: 0.628
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.630
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.632
    Metric valid/accuracy improved by 0.008 >= min_delta = 0.0. New best score: 0.640
    Metric valid/accuracy improved by 0.008 >= min_delta = 0.0. New best score: 0.648
    Metric valid/accuracy improved by 0.008 >= min_delta = 0.0. New best score: 0.656
    Metric valid/accuracy improved by 0.004 >= min_delta = 0.0. New best score: 0.660
    Metric valid/accuracy improved by 0.004 >= min_delta = 0.0. New best score: 0.664
    Metric valid/accuracy improved by 0.004 >= min_delta = 0.0. New best score: 0.668
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.670
    Metric valid/accuracy improved by 0.006 >= min_delta = 0.0. New best score: 0.676
    Metric valid/accuracy improved by 0.004 >= min_delta = 0.0. New best score: 0.680
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.682
    Metric valid/accuracy improved by 0.004 >= min_delta = 0.0. New best score: 0.686
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.688
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.690
    Metric valid/accuracy improved by 0.004 >= min_delta = 0.0. New best score: 0.694
    Metric valid/accuracy improved by 0.006 >= min_delta = 0.0. New best score: 0.700
    Metric valid/accuracy improved by 0.006 >= min_delta = 0.0. New best score: 0.706
    Metric valid/accuracy improved by 0.004 >= min_delta = 0.0. New best score: 0.710
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.712
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.714
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.716
    Metric valid/accuracy improved by 0.008 >= min_delta = 0.0. New best score: 0.724
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.726
    Metric valid/accuracy improved by 0.006 >= min_delta = 0.0. New best score: 0.732
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.734
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.736
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.738
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.740
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.742
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.744
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.746
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.748
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.750
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.752
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.754
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.756
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.758
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.760
    


.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>
    



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>
    


.. code:: ipython3

    trainer.validate(model, gloader, verbose=True,ckpt_path='best')#to restore best weights easily,we use trainer.validate and because we set the batchsize of dataloader as None,so the validate score may have some errors.we can use prediction step and compute the metrics


.. parsed-literal::

    Restoring states from the checkpoint path at /content/GCN_DGL_valid/accuracy=0.760000.ckpt
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    Loaded model weights from checkpoint at /content/GCN_DGL_valid/accuracy=0.760000.ckpt
    


.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃<span style="font-weight: bold">      Validate metric      </span>┃<span style="font-weight: bold">       DataLoader 0        </span>┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    │<span style="color: #008080; text-decoration-color: #008080">    test/accuracy_epoch    </span>│<span style="color: #800080; text-decoration-color: #800080">            0.0            </span>│
    │<span style="color: #008080; text-decoration-color: #008080">   valid/accuracy_epoch    </span>│<span style="color: #800080; text-decoration-color: #800080">            0.0            </span>│
    └───────────────────────────┴───────────────────────────┘
    </pre>
    



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>
    



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>
    




.. parsed-literal::

    [{'test/accuracy_epoch': 0.0, 'valid/accuracy_epoch': 0.0}]



.. code:: ipython3

    preds = trainer.predict(model,gloader)
    print(preds[0])
    print(preds[0].argmax(1))


.. parsed-literal::

    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    


.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>
    



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>
    


.. parsed-literal::

    tensor([[ 0.7378, -0.9543,  0.1648,  ...,  3.4618, -1.1093, -0.1241],
            [ 0.7513, -1.1883,  0.1382,  ...,  3.2373, -1.1376, -0.0872],
            [ 0.7513, -1.1883,  0.1382,  ...,  3.2373, -1.1376, -0.0872],
            ...,
            [ 0.8967, -0.0406,  0.6851,  ...,  1.0592, -0.6065, -0.4998],
            [ 0.3581, -0.6085, -0.3587,  ...,  1.0639,  0.7551,  0.9995],
            [ 0.3940, -1.2636,  0.3531,  ...,  1.3252,  0.1979,  0.9274]])
    tensor([4, 4, 4,  ..., 4, 3, 3])
    

