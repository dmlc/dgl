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
    Write your own GNN module
    =========================
    
    Sometimes, your model goes beyond simply stacking existing GNN modules.
    For example, you would like to invent a new way of aggregating neighbor
    information by considering node importance or edge weights.
    
    By the end of this tutorial you will be able to
    
    -  Understand DGL’s message passing APIs.
    -  Implement GraphSAGE convolution module by your own.
    
    This tutorial assumes that you already know :doc:`the basics of training a
    GNN for node classification <1_introduction>`.
    
    (Time estimate: 10 minutes)
    
    """
    
    import dgl
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    

.. code:: ipython3

    
    
    
    ######################################################################
    # Message passing and GNNs
    # ------------------------
    # 
    # DGL follows the *message passing paradigm* inspired by the Message
    # Passing Neural Network proposed by `Gilmer et
    # al. <https://arxiv.org/abs/1704.01212>`__ Essentially, they found many
    # GNN models can fit into the following framework:
    # 
    # .. math::
    # 
    # 
    #    m_{u\to v}^{(l)} = M^{(l)}\left(h_v^{(l-1)}, h_u^{(l-1)}, e_{u\to v}^{(l-1)}\right)
    # 
    # .. math::
    # 
    # 
    #    m_{v}^{(l)} = \sum_{u\in\mathcal{N}(v)}m_{u\to v}^{(l)}
    # 
    # .. math::
    # 
    # 
    #    h_v^{(l)} = U^{(l)}\left(h_v^{(l-1)}, m_v^{(l)}\right)
    # 
    # where DGL calls :math:`M^{(l)}` the *message function*, :math:`\sum` the
    # *reduce function* and :math:`U^{(l)}` the *update function*. Note that
    # :math:`\sum` here can represent any function and is not necessarily a
    # summation.
    # 
    
    
    ######################################################################
    # For example, the `GraphSAGE convolution (Hamilton et al.,
    # 2017) <https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf>`__
    # takes the following mathematical form:
    # 
    # .. math::
    # 
    # 
    #    h_{\mathcal{N}(v)}^k\leftarrow \text{Average}\{h_u^{k-1},\forall u\in\mathcal{N}(v)\}
    # 
    # .. math::
    # 
    # 
    #    h_v^k\leftarrow \text{ReLU}\left(W^k\cdot \text{CONCAT}(h_v^{k-1}, h_{\mathcal{N}(v)}^k) \right)
    # 
    # You can see that message passing is directional: the message sent from
    # one node :math:`u` to other node :math:`v` is not necessarily the same
    # as the other message sent from node :math:`v` to node :math:`u` in the
    # opposite direction.
    # 
    # Although DGL has builtin support of GraphSAGE via
    # :class:`dgl.nn.SAGEConv <dgl.nn.pytorch.SAGEConv>`,
    # here is how you can implement GraphSAGE convolution in DGL by your own.
    # 
    
    import dgl.function as fn
    
    
    class SAGEConv(nn.Module):
        """Graph convolution module used by the GraphSAGE model.
        
        Parameters
        ----------
        in_feat : int
            Input feature size.
        out_feat : int
            Output feature size.
        """
        def __init__(self, in_feat, out_feat):
            super(SAGEConv, self).__init__()
            # A linear submodule for projecting the input and neighbor feature to the output.
            self.linear = nn.Linear(in_feat * 2, out_feat)
        
        def forward(self, g, h):
            """Forward computation
            
            Parameters
            ----------
            g : Graph
                The input graph.
            h : Tensor
                The input node feature.
            """
            with g.local_scope():
                g.ndata['h'] = h
                # update_all is a message passing API.
                g.update_all(message_func=fn.copy_u('h', 'm'), reduce_func=fn.mean('m', 'h_N'))
                h_N = g.ndata['h_N']
                h_total = torch.cat([h, h_N], dim=1)
                return self.linear(h_total)
    
    
    ######################################################################
    # The central piece in this code is the
    # :func:`g.update_all <dgl.DGLGraph.update_all>`
    # function, which gathers and averages the neighbor features. There are
    # three concepts here:
    #
    # * Message function ``fn.copy_u('h', 'm')`` that
    #   copies the node feature under name ``'h'`` as *messages* sent to
    #   neighbors.
    #
    # * Reduce function ``fn.mean('m', 'h_N')`` that averages
    #   all the received messages under name ``'m'`` and saves the result as a
    #   new node feature ``'h_N'``.
    #
    # * ``update_all`` tells DGL to trigger the
    #   message and reduce functions for all the nodes and edges.
    # 
    
    
    ######################################################################
    # Afterwards, you can stack your own GraphSAGE convolution layers to form
    # a multi-layer GraphSAGE network.
    #
    
    class Model(nn.Module):
        def __init__(self, in_feats, h_feats, num_classes):
            super(Model, self).__init__()
            self.conv1 = SAGEConv(in_feats, h_feats)
            self.conv2 = SAGEConv(h_feats, num_classes)
        
        def forward(self, g, in_feat):
            h = self.conv1(g, in_feat)
            h = F.relu(h)
            h = self.conv2(g, h)
            return h
    
    
    ######################################################################
    # Training loop
    # ~~~~~~~~~~~~~
    # The following code for data loading and training loop is directly copied
    # from the introduction tutorial.
    # 
    
    
    
    class CustomSage(pl.LightningModule):
      def __init__(self,torch_model):
        super(CustomSage,self).__init__()
    
        self.model = torch_model
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        
      def forward(self,g,in_feat):
        logits = self.model(g, in_feat)
        return logits
    
      def training_step(self,batch,batch_idx):
        g = batch
        features = g.ndata['feat']
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        logits = self(g, features)
        pred = logits.argmax(1)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
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
    
    trainer = pl.Trainer(
        fast_dev_run=False,
        max_epochs=100,
        gpus=1,
        precision=32,
        check_val_every_n_epoch=1,
        val_check_interval=1.0,
        num_sanity_val_steps=0,
        callbacks=[
                   
        ModelCheckpoint(dirpath=os.getcwd(),filename=f'Graphsage_DGL_' + '{valid/accuracy:.6f}',monitor='valid/accuracy',mode='max',save_weights_only=False), \
        LearningRateMonitor(logging_interval='step'),EarlyStopping(monitor='valid/accuracy',min_delta=0.00,patience=30,verbose=True,mode='max',check_on_train_epoch_end=False),RichProgressBar()],
        logger=TensorBoardLogger(
        save_dir=os.getcwd(),
        version=1,
        name='Graphsage_by_DGL'
    )
      )
    
    from torch.utils.data import DataLoader
    import dgl.data
    dataset = dgl.data.CoraGraphDataset()
    g = dataset[0]
    
    gloader = DataLoader([g],batch_size=None) # use "[g]" instead of "g" 
    loader_for_valid = DataLoader([0],batch_size=None) ## use an empty daloader to enable validation_step of lightning model
    
    sagemodel = CustomSage(Model(g.ndata['feat'].shape[1], 16, dataset.num_classes))
    trainer.fit(sagemodel, gloader , loader_for_valid)
    
    
    preds = trainer.predict(sagemodel,gloader)
    print(preds[0])
    print(preds[0].argmax(1))


.. parsed-literal::

    GPU available: True, used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    `Trainer(val_check_interval=1.0)` was configured so validation will run at the end of the training epoch..
    

.. parsed-literal::

      NumNodes: 2708
      NumEdges: 10556
      NumFeats: 1433
      NumClasses: 7
      NumTrainingSamples: 140
      NumValidationSamples: 500
      NumTestSamples: 1000
    Done loading data from cached files.
    

.. parsed-literal::

    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    


.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┓
    ┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold">   </span>┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold"> Name           </span>┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold"> Type     </span>┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold"> Params </span>┃
    ┡━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━┩
    │<span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 0 </span>│ model          │ Model    │ 46.1 K │
    │<span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 1 </span>│ train_accuracy │ Accuracy │      0 │
    │<span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 2 </span>│ val_accuracy   │ Accuracy │      0 │
    │<span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 3 </span>│ test_accuracy  │ Accuracy │      0 │
    └───┴────────────────┴──────────┴────────┘
    </pre>
    



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Trainable params</span>: 46.1 K                                                                     
    <span style="font-weight: bold">Non-trainable params</span>: 0                                                                      
    <span style="font-weight: bold">Total params</span>: 46.1 K                                                                         
    <span style="font-weight: bold">Total estimated model params size (MB)</span>: 0                                                    
    </pre>
    



.. parsed-literal::

    Output()


.. parsed-literal::

    Metric valid/accuracy improved. New best score: 0.114
    Metric valid/accuracy improved by 0.004 >= min_delta = 0.0. New best score: 0.118
    Metric valid/accuracy improved by 0.134 >= min_delta = 0.0. New best score: 0.252
    Metric valid/accuracy improved by 0.264 >= min_delta = 0.0. New best score: 0.516
    Metric valid/accuracy improved by 0.070 >= min_delta = 0.0. New best score: 0.586
    Metric valid/accuracy improved by 0.052 >= min_delta = 0.0. New best score: 0.638
    Metric valid/accuracy improved by 0.016 >= min_delta = 0.0. New best score: 0.654
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.656
    Metric valid/accuracy improved by 0.004 >= min_delta = 0.0. New best score: 0.660
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.662
    Metric valid/accuracy improved by 0.010 >= min_delta = 0.0. New best score: 0.672
    Metric valid/accuracy improved by 0.016 >= min_delta = 0.0. New best score: 0.688
    Metric valid/accuracy improved by 0.004 >= min_delta = 0.0. New best score: 0.692
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.694
    Metric valid/accuracy improved by 0.006 >= min_delta = 0.0. New best score: 0.700
    Metric valid/accuracy improved by 0.004 >= min_delta = 0.0. New best score: 0.704
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.706
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.708
    Metric valid/accuracy improved by 0.004 >= min_delta = 0.0. New best score: 0.712
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.714
    Metric valid/accuracy improved by 0.004 >= min_delta = 0.0. New best score: 0.718
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.720
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.722
    Metric valid/accuracy improved by 0.008 >= min_delta = 0.0. New best score: 0.730
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.732
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.734
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.736
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.738
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.740
    Metric valid/accuracy improved by 0.004 >= min_delta = 0.0. New best score: 0.744
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.746
    Monitored metric valid/accuracy did not improve in the last 30 records. Best score: 0.746. Signaling Trainer to stop.
    


.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>
    



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>
    


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

    tensor([[-1.3196, -0.3394, -0.9310,  ...,  3.9556, -2.1971, -0.3560],
            [-0.6079, -1.0620, -0.6462,  ...,  3.4164, -2.4671,  0.2075],
            [-1.5185, -0.4258, -0.5863,  ...,  4.0568, -2.3809,  0.2578],
            ...,
            [ 0.5408, -0.8290, -0.4664,  ...,  0.6450, -1.2075, -0.7228],
            [-1.0078, -1.7803, -0.0888,  ..., -0.1327,  0.6172, -1.9845],
            [ 0.0656, -2.0408,  0.4820,  ..., -0.3377, -0.1744, -2.2688]])
    tensor([4, 4, 4,  ..., 4, 3, 3])
    

.. code:: ipython3

    
    
    
    ######################################################################
    # More customization
    # ------------------
    # 
    # In DGL, we provide many built-in message and reduce functions under the
    # ``dgl.function`` package. You can find more details in :ref:`the API
    # doc <apifunction>`.
    # 
    
    
    ######################################################################
    # These APIs allow one to quickly implement new graph convolution modules.
    # For example, the following implements a new ``SAGEConv`` that aggregates
    # neighbor representations using a weighted average. Note that ``edata``
    # member can hold edge features which can also take part in message
    # passing.
    # 
    
    class WeightedSAGEConv(nn.Module):
        """Graph convolution module used by the GraphSAGE model with edge weights.
        
        Parameters
        ----------
        in_feat : int
            Input feature size.
        out_feat : int
            Output feature size.
        """
        def __init__(self, in_feat, out_feat):
            super(WeightedSAGEConv, self).__init__()
            # A linear submodule for projecting the input and neighbor feature to the output.
            self.linear = nn.Linear(in_feat * 2, out_feat)
        
        def forward(self, g, h, w):
            """Forward computation
            
            Parameters
            ----------
            g : Graph
                The input graph.
            h : Tensor
                The input node feature.
            w : Tensor
                The edge weight.
            """
            with g.local_scope():
                g.ndata['h'] = h
                g.edata['w'] = w
                g.update_all(message_func=fn.u_mul_e('h', 'w', 'm'), reduce_func=fn.mean('m', 'h_N'))
                h_N = g.ndata['h_N']
                h_total = torch.cat([h, h_N], dim=1)
                return self.linear(h_total)
    
    
    ######################################################################
    # Because the graph in this dataset does not have edge weights, we
    # manually assign all edge weights to one in the ``forward()`` function of
    # the model. You can replace it with your own edge weights.
    # 
    
    class AnotherModel(nn.Module):
        def __init__(self, in_feats, h_feats, num_classes):
            super(AnotherModel, self).__init__()
            self.conv1 = WeightedSAGEConv(in_feats, h_feats)
            self.conv2 = WeightedSAGEConv(h_feats, num_classes)
        
        def forward(self, g, in_feat):
            h = self.conv1(g, in_feat, torch.ones(g.num_edges(), 1).to(g.device))
            h = F.relu(h)
            h = self.conv2(g, h, torch.ones(g.num_edges(), 1).to(g.device))
            return h
        
    gloader = DataLoader([g],batch_size=None) # use "[g]" instead of "g" 
    loader_for_valid = DataLoader([0],batch_size=None) ## use an empty daloader to enable validation_step of lightning model
    
    weightsagemodel = CustomSage(AnotherModel(g.ndata['feat'].shape[1], 16, dataset.num_classes))
    
    trainer = pl.Trainer(
        fast_dev_run=False,
        max_epochs=100,
        gpus=1,
        precision=32,
        check_val_every_n_epoch=1,
        val_check_interval=1.0,
        num_sanity_val_steps=0,
        callbacks=[
                   
        ModelCheckpoint(dirpath=os.getcwd(),filename=f'Graphsage_DGL_' + '{valid/accuracy:.6f}',monitor='valid/accuracy',mode='max',save_weights_only=False), \
        LearningRateMonitor(logging_interval='step'),EarlyStopping(monitor='valid/accuracy',min_delta=0.00,patience=30,verbose=True,mode='max',check_on_train_epoch_end=False),RichProgressBar()],
        logger=TensorBoardLogger(
        save_dir=os.getcwd(),
        version=1,
        name='Graphsage_by_DGL'
    )
      )
    
    
    trainer.fit(weightsagemodel, gloader , loader_for_valid)
    
    
    preds = trainer.predict(weightsagemodel,gloader)
    print(preds[0])
    print(preds[0].argmax(1))
    
    


.. parsed-literal::

    GPU available: True, used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    `Trainer(val_check_interval=1.0)` was configured so validation will run at the end of the training epoch..
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    


.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━┓
    ┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold">   </span>┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold"> Name           </span>┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold"> Type         </span>┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold"> Params </span>┃
    ┡━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━┩
    │<span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 0 </span>│ model          │ AnotherModel │ 46.1 K │
    │<span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 1 </span>│ train_accuracy │ Accuracy     │      0 │
    │<span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 2 </span>│ val_accuracy   │ Accuracy     │      0 │
    │<span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 3 </span>│ test_accuracy  │ Accuracy     │      0 │
    └───┴────────────────┴──────────────┴────────┘
    </pre>
    



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Trainable params</span>: 46.1 K                                                                     
    <span style="font-weight: bold">Non-trainable params</span>: 0                                                                      
    <span style="font-weight: bold">Total params</span>: 46.1 K                                                                         
    <span style="font-weight: bold">Total estimated model params size (MB)</span>: 0                                                    
    </pre>
    



.. parsed-literal::

    Output()


.. parsed-literal::

    Metric valid/accuracy improved. New best score: 0.114
    Metric valid/accuracy improved by 0.004 >= min_delta = 0.0. New best score: 0.118
    Metric valid/accuracy improved by 0.054 >= min_delta = 0.0. New best score: 0.172
    Metric valid/accuracy improved by 0.138 >= min_delta = 0.0. New best score: 0.310
    Metric valid/accuracy improved by 0.094 >= min_delta = 0.0. New best score: 0.404
    Metric valid/accuracy improved by 0.012 >= min_delta = 0.0. New best score: 0.416
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.418
    Metric valid/accuracy improved by 0.004 >= min_delta = 0.0. New best score: 0.422
    Metric valid/accuracy improved by 0.004 >= min_delta = 0.0. New best score: 0.426
    Metric valid/accuracy improved by 0.004 >= min_delta = 0.0. New best score: 0.430
    Metric valid/accuracy improved by 0.014 >= min_delta = 0.0. New best score: 0.444
    Metric valid/accuracy improved by 0.020 >= min_delta = 0.0. New best score: 0.464
    Metric valid/accuracy improved by 0.008 >= min_delta = 0.0. New best score: 0.472
    Metric valid/accuracy improved by 0.008 >= min_delta = 0.0. New best score: 0.480
    Metric valid/accuracy improved by 0.006 >= min_delta = 0.0. New best score: 0.486
    Metric valid/accuracy improved by 0.006 >= min_delta = 0.0. New best score: 0.492
    Metric valid/accuracy improved by 0.004 >= min_delta = 0.0. New best score: 0.496
    Metric valid/accuracy improved by 0.016 >= min_delta = 0.0. New best score: 0.512
    Metric valid/accuracy improved by 0.008 >= min_delta = 0.0. New best score: 0.520
    Metric valid/accuracy improved by 0.014 >= min_delta = 0.0. New best score: 0.534
    Metric valid/accuracy improved by 0.018 >= min_delta = 0.0. New best score: 0.552
    Metric valid/accuracy improved by 0.008 >= min_delta = 0.0. New best score: 0.560
    Metric valid/accuracy improved by 0.016 >= min_delta = 0.0. New best score: 0.576
    Metric valid/accuracy improved by 0.026 >= min_delta = 0.0. New best score: 0.602
    Metric valid/accuracy improved by 0.010 >= min_delta = 0.0. New best score: 0.612
    Metric valid/accuracy improved by 0.008 >= min_delta = 0.0. New best score: 0.620
    Metric valid/accuracy improved by 0.022 >= min_delta = 0.0. New best score: 0.642
    Metric valid/accuracy improved by 0.016 >= min_delta = 0.0. New best score: 0.658
    Metric valid/accuracy improved by 0.014 >= min_delta = 0.0. New best score: 0.672
    Metric valid/accuracy improved by 0.012 >= min_delta = 0.0. New best score: 0.684
    Metric valid/accuracy improved by 0.012 >= min_delta = 0.0. New best score: 0.696
    Metric valid/accuracy improved by 0.012 >= min_delta = 0.0. New best score: 0.708
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.710
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.712
    Metric valid/accuracy improved by 0.010 >= min_delta = 0.0. New best score: 0.722
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.724
    Metric valid/accuracy improved by 0.008 >= min_delta = 0.0. New best score: 0.732
    Metric valid/accuracy improved by 0.004 >= min_delta = 0.0. New best score: 0.736
    Metric valid/accuracy improved by 0.004 >= min_delta = 0.0. New best score: 0.740
    Metric valid/accuracy improved by 0.006 >= min_delta = 0.0. New best score: 0.746
    Metric valid/accuracy improved by 0.006 >= min_delta = 0.0. New best score: 0.752
    Metric valid/accuracy improved by 0.004 >= min_delta = 0.0. New best score: 0.756
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.758
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.760
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.762
    Metric valid/accuracy improved by 0.002 >= min_delta = 0.0. New best score: 0.764
    Monitored metric valid/accuracy did not improve in the last 30 records. Best score: 0.764. Signaling Trainer to stop.
    


.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>
    



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>
    


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

    tensor([[-0.8000, -1.1534, -2.4853,  ...,  4.0224, -2.0878, -1.1865],
            [-0.4220, -1.8924, -2.2138,  ...,  3.7749, -2.1546, -0.7663],
            [-0.5933, -1.3474, -2.3761,  ...,  3.9435, -2.4111, -0.7804],
            ...,
            [ 0.7434, -1.4427, -0.7403,  ...,  0.8430, -1.0685, -1.8600],
            [-2.2793, -0.6009, -1.3736,  ..., -1.2937,  0.8173, -2.8112],
            [-1.1879, -0.5219, -1.2123,  ..., -1.2102,  0.3441, -3.1043]])
    tensor([4, 4, 4,  ..., 4, 3, 3])
    

.. code:: ipython3

    
    ######################################################################
    # Even more customization by user-defined function
    # ------------------------------------------------
    # 
    # DGL allows user-defined message and reduce function for the maximal
    # expressiveness. Here is a user-defined message function that is
    # equivalent to ``fn.u_mul_e('h', 'w', 'm')``.
    # 
    
    def u_mul_e_udf(edges):
        return {'m' : edges.src['h'] * edges.data['w']}
    
    
    ######################################################################
    # ``edges`` has three members: ``src``, ``data`` and ``dst``, representing
    # the source node feature, edge feature, and destination node feature for
    # all edges.
    # 
    
    
    ######################################################################
    # You can also write your own reduce function. For example, the following
    # is equivalent to the builtin ``fn.mean('m', 'h_N')`` function that averages
    # the incoming messages:
    # 
    
    def mean_udf(nodes):
        return {'h_N': nodes.mailbox['m'].mean(1)}
    
    
    ######################################################################
    # In short, DGL will group the nodes by their in-degrees, and for each
    # group DGL stacks the incoming messages along the second dimension. You 
    # can then perform a reduction along the second dimension to aggregate
    # messages.
    # 
    # For more details on customizing message and reduce function with
    # user-defined function, please refer to the :ref:`API
    # reference <apiudf>`.
    # 
    
    
    ######################################################################
    # Best practice of writing custom GNN modules
    # -------------------------------------------
    # 
    # DGL recommends the following practice ranked by preference:
    # 
    # -  Use ``dgl.nn`` modules.
    # -  Use ``dgl.nn.functional`` functions which contain lower-level complex
    #    operations such as computing a softmax for each node over incoming
    #    edges.
    # -  Use ``update_all`` with builtin message and reduce functions.
    # -  Use user-defined message or reduce functions.
    # 
    
    
    ######################################################################
    # What’s next?
    # ------------
    # 
    # -  :ref:`Writing Efficient Message Passing
    #    Code <guide-message-passing-efficient>`.
    # 
    
    
    # Thumbnail credits: Representation Learning on Networks, Jure Leskovec, WWW 2018
    # sphinx_gallery_thumbnail_path = '_static/blitz_3_message_passing.png'
    

.. code:: ipython3

    ######################################################################
    # A more complex example
    


