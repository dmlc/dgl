.. code:: python

    # !pip install pytorch-lightning
    # !pip install torchlayers
    # !pip install torchmetric
    # !pip install dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html
    #!pip install rich --upgrade

.. code:: python

    """
    Link Prediction using Graph Neural Networks
    ===========================================
    
    In the :doc:`introduction <1_introduction>`, you have already learned
    the basic workflow of using GNNs for node classification,
    i.e. predicting the category of a node in a graph. This tutorial will
    teach you how to train a GNN for link prediction, i.e. predicting the
    existence of an edge between two arbitrary nodes in a graph.
    
    By the end of this tutorial you will be able to
    
    -  Build a GNN-based link prediction model.
    -  Train and evaluate the model on a small DGL-provided dataset.
    
    (Time estimate: 28 minutes)
    
    """
    
    import pytorch_lightning as pl
    import torchmetrics
    import dgl
    import warnings
    warnings.filterwarnings('ignore')
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    
    import itertools
    import numpy as np
    import scipy.sparse as sp

.. code:: python

    
    
    ######################################################################
    # Overview of Link Prediction with GNN
    # ------------------------------------
    #
    # Many applications such as social recommendation, item recommendation,
    # knowledge graph completion, etc., can be formulated as link prediction,
    # which predicts whether an edge exists between two particular nodes. This
    # tutorial shows an example of predicting whether a citation relationship,
    # either citing or being cited, between two papers exists in a citation
    # network.
    #
    # This tutorial formulates the link prediction problem as a binary classification
    # problem as follows:
    #
    # -  Treat the edges in the graph as *positive examples*.
    # -  Sample a number of non-existent edges (i.e. node pairs with no edges
    #    between them) as *negative* examples.
    # -  Divide the positive examples and negative examples into a training
    #    set and a test set.
    # -  Evaluate the model with any binary classification metric such as Area
    #    Under Curve (AUC).
    #
    # .. note::
    #
    #    The practice comes from
    #    `SEAL <https://papers.nips.cc/paper/2018/file/53f0d7c537d99b3824f0f99d62ea2428-Paper.pdf>`__,
    #    although the model here does not use their idea of node labeling.
    #
    # In some domains such as large-scale recommender systems or information
    # retrieval, you may favor metrics that emphasize good performance of
    # top-K predictions. In these cases you may want to consider other metrics
    # such as mean average precision, and use other negative sampling methods,
    # which are beyond the scope of this tutorial.
    #
    # Loading graph and features
    # --------------------------
    #
    # Following the :doc:`introduction <1_introduction>`, this tutorial
    # first loads the Cora dataset.
    #
    
    import dgl.data
    
    dataset = dgl.data.CoraGraphDataset()
    g = dataset[0]
    
    
    ######################################################################
    # Prepare training and testing sets
    # ---------------------------------
    #
    # This tutorial randomly picks 10% of the edges for positive examples in
    # the test set, and leave the rest for the training set. It then samples
    # the same number of edges for negative examples in both sets.
    #
    
    # Split edge set for training and testing
    u, v = g.edges()
    
    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.1)
    train_size = g.number_of_edges() - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]
    
    # Find all negative edges and split them for training and testing
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)
    
    neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]
    
    
    ######################################################################
    # When training, you will need to remove the edges in the test set from
    # the original graph. You can do this via ``dgl.remove_edges``.
    #
    # .. note::
    #
    #    ``dgl.remove_edges`` works by creating a subgraph from the
    #    original graph, resulting in a copy and therefore could be slow for
    #    large graphs. If so, you could save the training and test graph to
    #    disk, as you would do for preprocessing.
    #
    
    
    


.. parsed-literal::

      NumNodes: 2708
      NumEdges: 10556
      NumFeats: 1433
      NumClasses: 7
      NumTrainingSamples: 140
      NumValidationSamples: 500
      NumTestSamples: 1000
    Done loading data from cached files.
    

.. code:: python

    
    ######################################################################
    # Define a GraphSAGE model
    # ------------------------
    #
    # This tutorial builds a model consisting of two
    # `GraphSAGE <https://arxiv.org/abs/1706.02216>`__ layers, each computes
    # new node representations by averaging neighbor information. DGL provides
    # ``dgl.nn.SAGEConv`` that conveniently creates a GraphSAGE layer.
    #
    
    from dgl.nn import SAGEConv
    
    # ----------- 2. create model -------------- #
    # build a two-layer GraphSAGE model
    class GraphSAGE(nn.Module):
        def __init__(self, in_feats, h_feats):
            super(GraphSAGE, self).__init__()
            self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
            self.conv2 = SAGEConv(h_feats, h_feats, 'mean')
        
        def forward(self, g, in_feat):
            h = self.conv1(g, in_feat)
            h = F.relu(h)
            h = self.conv2(g, h)
            return h

.. code:: python

    
    ######################################################################
    # The model then predicts the probability of existence of an edge by
    # computing a score between the representations of both incident nodes
    # with a function (e.g. an MLP or a dot product), which you will see in
    # the next section.
    #
    # .. math::
    #
    #
    #    \hat{y}_{u\sim v} = f(h_u, h_v)
    #
    
    
    ######################################################################
    # Positive graph, negative graph, and ``apply_edges``
    # ---------------------------------------------------
    #
    # In previous tutorials you have learned how to compute node
    # representations with a GNN. However, link prediction requires you to
    # compute representation of *pairs of nodes*.
    #
    # DGL recommends you to treat the pairs of nodes as another graph, since
    # you can describe a pair of nodes with an edge. In link prediction, you
    # will have a *positive graph* consisting of all the positive examples as
    # edges, and a *negative graph* consisting of all the negative examples.
    # The *positive graph* and the *negative graph* will contain the same set
    # of nodes as the original graph.  This makes it easier to pass node
    # features among multiple graphs for computation.  As you will see later,
    # you can directly feed the node representations computed on the entire
    # graph to the positive and the negative graphs for computing pair-wise
    # scores.
    #
    # The following code constructs the positive graph and the negative graph
    # for the training set and the test set respectively.
    #
    
    train_g = dgl.remove_edges(g, eids[:test_size])
    
    
    
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes()) # we only need the topology information 
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())
    
    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

.. code:: python

    device =  'cuda' if torch.cuda.is_available() else 'cpu'
    
    class Sage4LinkPrediction(pl.LightningModule):
      def __init__(self,gnn,predictor,device):
        super().__init__()
        self.gnn = gnn
        self.pred = predictor
        self.train_auc = torchmetrics.AUROC(pos_label=1)
        self.val_auc = torchmetrics.AUROC(pos_label=1)
        self.automatic_optimization = True
        self.save_hyperparameters('device') # save the hyperparams to model.params
    
    
      def forward(self,g):
        h = self.gnn(g, g.ndata['feat'])
        
        return h
    
      def training_step(self,batch,batch_idx):
        train_g,train_pos_g,train_neg_g = batch
        h = self(train_g)
        pos_score = self.pred(train_pos_g, h)
        neg_score = self.pred(train_neg_g, h)
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([torch.ones(pos_score.shape[0]).to(self.hparams.device), torch.zeros(neg_score.shape[0]).to(self.hparams.device)]) # make sure all tensors to be on the same device
        # a better way is to use register buffer in "__init__" or define the labels in dataloader
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        self.train_auc(scores,labels.long())
        self.log_dict({'train-loss':loss,'train-auc':self.train_auc},prog_bar=True,on_step=True,on_epoch=True,batch_size=1)
        self.h = h # for validation
        return loss
    
      def validation_step(self,batch,batch_idx):
        test_pos_g,test_neg_g = batch
        pos_score = self.pred(test_pos_g, self.h)
        neg_score = self.pred(test_neg_g, self.h)
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([torch.ones(pos_score.shape[0]).to(self.hparams.device), torch.zeros(neg_score.shape[0]).to(self.hparams.device)])
        self.val_auc(scores,labels.long())
    
      def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=1e-2,weight_decay=1e-7)
        return [optimizer]
    
      def prediction_step(self,batch,batch_idx):
        g=batch
        return self(g)
    

.. code:: python

    
    
    ######################################################################
    # The benefit of treating the pairs of nodes as a graph is that you can
    # use the ``DGLGraph.apply_edges`` method, which conveniently computes new
    # edge features based on the incident nodes’ features and the original
    # edge features (if applicable).
    #
    # DGL provides a set of optimized builtin functions to compute new
    # edge features based on the original node/edge features. For example,
    # ``dgl.function.u_dot_v`` computes a dot product of the incident nodes’
    # representations for each edge.
    #
    
    import dgl.function as fn
    
    class DotPredictor(nn.Module):
        def forward(self, g, h):
            with g.local_scope():
                g.ndata['h'] = h
                # Compute a new edge feature named 'score' by a dot-product between the
                # source node feature 'h' and destination node feature 'h'.
                g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
                # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
                return g.edata['score'][:, 0]
    
    
    ######################################################################
    # You can also write your own function if it is complex.
    # For instance, the following module produces a scalar score on each edge
    # by concatenating the incident nodes’ features and passing it to an MLP.
    #
    
    class MLPPredictor(nn.Module):
        def __init__(self, h_feats):
            super().__init__()
            self.W1 = nn.Linear(h_feats * 2, h_feats)
            self.W2 = nn.Linear(h_feats, 1)
    
        def apply_edges(self, edges):
            """
            Computes a scalar score for each edge of the given graph.
    
            Parameters
            ----------
            edges :
                Has three members ``src``, ``dst`` and ``data``, each of
                which is a dictionary representing the features of the
                source nodes, the destination nodes, and the edges
                themselves.
    
            Returns
            -------
            dict
                A dictionary of new edge features.
            """
            h = torch.cat([edges.src['h'], edges.dst['h']], 1)
            return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}
    
        def forward(self, g, h):
            with g.local_scope():
                g.ndata['h'] = h
                g.apply_edges(self.apply_edges)
                return g.edata['score']
    
    
    ######################################################################
    # .. note::
    #
    #    The builtin functions are optimized for both speed and memory.
    #    We recommend using builtin functions whenever possible.
    #
    # .. note::
    #
    #    If you have read the :doc:`message passing
    #    tutorial <3_message_passing>`, you will notice that the
    #    argument ``apply_edges`` takes has exactly the same form as a message
    #    function in ``update_all``.
    #
    
    
    ######################################################################
    # Training loop
    # -------------
    #
    # After you defined the node representation computation and the edge score
    # computation, you can go ahead and define the overall model, loss
    # function, and evaluation metric.
    #
    # The loss function is simply binary cross entropy loss.
    #
    # .. math::
    #
    #
    #    \mathcal{L} = -\sum_{u\sim v\in \mathcal{D}}\left( y_{u\sim v}\log(\hat{y}_{u\sim v}) + (1-y_{u\sim v})\log(1-\hat{y}_{u\sim v})) \right)
    #
    # The evaluation metric in this tutorial is AUC.
    #
    
    
    
    
    
    

.. code:: python

    ######################################################################
    # The training loop goes as follows:
    #
    # .. note::
    #
    #    This tutorial does not include evaluation on a validation
    #    set. In practice you should save and evaluate the best model based on
    #    performance on the validation set.
    #
    
    # ----------- 3. set up loss and optimizer -------------- #
    # in this case, loss will in training loop
    
    # ----------- 4. training -------------------------------- #
    from torch.utils.data import DataLoader
    train_g_dataloader =DataLoader([[train_g,train_pos_g,train_neg_g]],batch_size=None)
    test_g_dataloader =DataLoader([[test_pos_g,test_neg_g]],batch_size=None)
    model = GraphSAGE(train_g.ndata['feat'].shape[1], 16)
    pred = DotPredictor()
    
    sage4linkpred = Sage4LinkPrediction(model,pred,device)
    
    from pytorch_lightning.callbacks import RichProgressBar
    trainer = pl.Trainer(
      fast_dev_run=False,
      max_epochs=10,
      gpus=1,
      precision=32,
      check_val_every_n_epoch=1,
      val_check_interval=1.0,
      num_sanity_val_steps=0,
      callbacks=[RichProgressBar()],
    )
    trainer.fit(sage4linkpred,train_g_dataloader,test_g_dataloader)
    
    
    # Thumbnail credits: Link Prediction with Neo4j, Mark Needham
    # sphinx_gallery_thumbnail_path = '_static/blitz_4_link_predict.png'


.. parsed-literal::

    GPU available: True, used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    `Trainer(val_check_interval=1.0)` was configured so validation will run at the end of the training epoch..
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    


.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━┓
    ┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold">   </span>┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold"> Name      </span>┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold"> Type         </span>┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold"> Params </span>┃
    ┡━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━┩
    │<span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 0 </span>│ gnn       │ GraphSAGE    │ 46.4 K │
    │<span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 1 </span>│ pred      │ DotPredictor │      0 │
    │<span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 2 </span>│ train_auc │ AUROC        │      0 │
    │<span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 3 </span>│ val_auc   │ AUROC        │      0 │
    └───┴───────────┴──────────────┴────────┘
    </pre>
    



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Trainable params</span>: 46.4 K                                                                     
    <span style="font-weight: bold">Non-trainable params</span>: 0                                                                      
    <span style="font-weight: bold">Total params</span>: 46.4 K                                                                         
    <span style="font-weight: bold">Total estimated model params size (MB)</span>: 0                                                    
    </pre>
    



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>
    



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>
    


