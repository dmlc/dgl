##  Overview

This project demonstrates the training and evaluation of a GraphSAGE model for node classification on large graphs. The example utilizes GraphBolt for efficient data handling and PyG for the GNN training.

##  Requirements

DGL with GraphBolt
PyTorch Geometric
torchmetrics
Dataset
The ogbn-arxiv dataset is used, loaded through GraphBolt's BuiltinDataset.

##  Model
The model is a three-layer GraphSAGE network implemented using PyTorch Geometric's SAGEConv layers.

##  Training
##  Training is performed with the following settings:

Mini-batch size: 1024
Neighbor sampling: [10, 10, 10]
Optimizer: Ada
Learning Rate: 0.01
Weight Decay: 5e-4
Loss Function: CrossEntropyLoss
Evaluation is done separately for validation and test datasets. The model's performance is measured in terms of accuracy using the torchmetrics.functional library.

##  How to Run
Ensure all required libraries are installed.
Run the main function in the provided script.

##  Results
The results include training loss, training accuracy, validation accuracy, and test accuracy for each epoch. Here's an example of the output format:


Epoch 0, Train Loss: 2.1434143031581065, Train Accuracy: 0.40741799628330455, Valid Accuracy: 0.5174334645271301, Test Accuracy: 0.45931321382522583
Epoch 1, Train Loss: 1.5044617867201902, Train Accuracy: 0.5599014745824216, Valid Accuracy: 0.5585758090019226, Test Accuracy: 0.5000720024108887
Epoch 2, Train Loss: 1.4136397892169739, Train Accuracy: 0.5799584345894591, Valid Accuracy: 0.5738447308540344, Test Accuracy: 0.5212023854255676
Epoch 3, Train Loss: 1.3619167322523138, Train Accuracy: 0.5936926138925237, Valid Accuracy: 0.5808584094047546, Test Accuracy: 0.5153385400772095
Epoch 4, Train Loss: 1.332796195919594, Train Accuracy: 0.6019617114392848, Valid Accuracy: 0.5746166110038757, Test Accuracy: 0.5160381197929382
Epoch 5, Train Loss: 1.3136085218258118, Train Accuracy: 0.6065031174057905, Valid Accuracy: 0.5799188017845154, Test Accuracy: 0.5128901600837708
Epoch 6, Train Loss: 1.2895360662696067, Train Accuracy: 0.6135406472328213, Valid Accuracy: 0.5840128660202026, Test Accuracy: 0.5215933322906494
Epoch 7, Train Loss: 1.2913014500328663, Train Accuracy: 0.613221759162534, Valid Accuracy: 0.5814624428749084, Test Accuracy: 0.5052363276481628
Epoch 8, Train Loss: 1.267129360959771, Train Accuracy: 0.6194345784629595, Valid Accuracy: 0.6039800047874451, Test Accuracy: 0.5395757555961609
Epoch 9, Train Loss: 1.2580654018380668, Train Accuracy: 0.6219856830252581, Valid Accuracy: 0.5980402231216431, Test Accuracy: 0.5351521372795105

##  Observations
Training Time: Provide an average duration per epoch or total training time.
Accuracy Trends: Highlight how accuracy evolves over epochs for training, validation, and test sets.
Performance Insights: Any insights or observations regarding the model's performance, training stability, etc.
