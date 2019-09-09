# Junction Tree Variational Autoencoder for Molecular Graph Generation (JTNN)

Wengong Jin, Regina Barzilay, Tommi Jaakkola. 
Junction Tree Variational Autoencoder for Molecular Graph Generation. 
*arXiv preprint arXiv:1802.04364*, 2018.

JTNN uses algorithm called junction tree algorithm to form a tree from the molecular graph. 
Then the model will encode the tree and graph into two separate vectors `z_G` and `z_T`. Details can
be found in original paper. The brief process is as below (from original paper): 

![image](https://user-images.githubusercontent.com/8686776/63677300-3fb6d980-c81f-11e9-8a65-57c8b03aaf52.png)

**Goal**: JTNN is an auto-encoder model, aiming to learn hidden representation for molecular graphs. 
These representations can be used for downstream tasks, such as property prediction, or molecule optimizations.

## Dataset

### ZINC

> The ZINC database is a curated collection of commercially available chemical compounds 
prepared especially for virtual screening. (introduction from Wikipedia)

Generally speaking, molecules in the ZINC dataset are more drug-like. We uses ~220,000 
molecules for training and 5000 molecules for validation. 

### Preprocessing

Class `JTNNDataset` will process a SMILES into a dict, including the junction tree, graph with 
encoded nodes(atoms) and edges(bonds), and other information for model to use.

## Usage

### Training

To start training, use `python train.py`. By default, the script will use ZINC dataset
 with preprocessed vocabulary, and save model checkpoint at the current working directory. 
```
-s SAVE_PATH, Path to save checkpoint models, default to be current
              working directory (default: ./)
-m MODEL_PATH, Path to load pre-trained model (default: None)
-b BATCH_SIZE, Batch size (default: 40)
-w HIDDEN_SIZE, Size of representation vectors (default: 200)
-l LATENT_SIZE, Latent Size of node(atom) features and edge(atom)
                features (default: 56)
-d DEPTH, Depth of message passing hops (default: 3)
-z BETA, Coefficient of KL Divergence term (default: 1.0)
-q LR, Learning Rate (default: 0.001)
```

Model will be saved periodically. 
All training checkpoint will be stored at `SAVE_PATH`, passed by command line or by default.

#### Dataset configuration

If you want to use your own dataset, please create a file contains one SMILES a line, 
 and pass the file path to the `-t` or `--train` option.
```
  -t TRAIN, --train TRAIN
                        Training file name (default: train)
```

### Evaluation

To start evaluation, use `python reconstruct_eval.py`, and following arguments
```
-t TRAIN, Training file name (default: test)
-m MODEL_PATH, Pre-trained model to be loaded for evalutaion. If not
               specified, would use pre-trained model from model zoo
               (default: None)
-w HIDDEN_SIZE, Hidden size of representation vector, should be
                consistent with pre-trained model (default: 450)
-l LATENT_SIZE, Latent Size of node(atom) features and edge(atom)
                features, should be consistent with pre-trained model
                 (default: 56)
-d DEPTH, Depth of message passing hops, should be consistent
          with pre-trained model (default: 3)
```

And it would print out the success rate of reconstructing the same molecules.

### Pre-trained models

Below gives the statistics of pre-trained `JTNN_ZINC` model. 

| Pre-trained model  | % Reconstruction Accuracy
| ------------------ | -------
| `JTNN_ZINC`        |  73.7             

### Visualization

Here we draw some "neighbor" of a given molecule, by adding noises on the intermediate representations. Detailed script can be found [here](https://s3.us-east-2.amazonaws.com/dgl.ai/model_zoo/drug_discovery/jtnn/viz_neighbor_mol.ipynb). Please put this script at the current directory (`examples/pytorch/model_zoo/chem/generative_models/jtnn/`).

#### Given Molecule
![image](https://user-images.githubusercontent.com/8686776/63773593-0d37da00-c90e-11e9-8933-0abca4b430db.png)
#### Neighbor Molecules
![image](https://user-images.githubusercontent.com/8686776/63773602-1163f780-c90e-11e9-8341-5122dc0d0c82.png)

### Warnings from PyTorch 1.2
If you are using PyTorch 1.2, there might be warning saying 
`UserWarning: indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead.`. This is due to the new feature in PyTorch 1.2. Please kindly ignore it.