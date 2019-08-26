# Junction Tree Variational Autoencoder for Molecular Graph Generation (JTNN)

Wengong Jin, Regina Barzilay, Tommi Jaakkola. 
Junction Tree Variational Autoencoder for Molecular Graph Generation. 
*arXiv preprint arXiv:1802.04364*, 2018.

JTNN uses algorithm called junction tree algorithm to form a tree from the molecule graph. 
Then the model will encode the tree and graph into two separate vectors `z_G` and `z_T`. Details can
be found in original paper. The brief process is as below (from original paper):
![image](https://user-images.githubusercontent.com/8686776/63677300-3fb6d980-c81f-11e9-8a65-57c8b03aaf52.png)


**Goal**: JTNN is an auto-encoder model, aiming to learn hidden representation for molecule graphs. 
These representations can be used for downstream task, such as property prediction, or molecule optimizations.

## Dataset

### ZINC

> The ZINC database is a curated collection of commercially available chemical compounds 
prepared especially for virtual screening. (introduction from Wikipedia)

Generally speaking, molecules in ZINC dataset are more drug-like molecules. we uses ~220,000 
molecules for training and 5000 molecules for validation. 

### Preprocessing

Class `JTNNDataset` will process a smile into a dict, including the junction tree, graph with 
encoded nodes(atoms) and edges(bonds), and other information for model to use.

## Usage

### Training

To start training, use `python train.py`. By default, the script will use ZINC dataset
 with preprocessed vocabulary, and saving model checkpoint at the current working directory. 
```
  -s SAVE_PATH, --save_dir SAVE_PATH
                        Path to save checkpoint models, default to be current
                        working directory (default: ./)
  -m MODEL_PATH, --model MODEL_PATH
                        Path to load pre-trained model (default: None)
  -b BATCH_SIZE, --batch BATCH_SIZE
                        Batch size (default: 40)
  -w HIDDEN_SIZE, --hidden HIDDEN_SIZE
                        Size of representation vectors (default: 200)
  -l LATENT_SIZE, --latent LATENT_SIZE
                        Latent Size of node(atom) features and edge(atom)
                        features (default: 56)
  -d DEPTH, --depth DEPTH
                        Depth of message passing hops (default: 3)
  -z BETA, --beta BETA  Coefficient of KL Divergence term (default: 1.0)
  -q LR, --lr LR        Learning Rate (default: 0.001)
  -T, --test            Add this flag to run test mode (default: False)
```

Model will be saved periodically. 
All training checkpoint will be stored at `SAVE_PATH`, passed by command line or by default.

#### Dataset configuration

If you want to use your own dataset, please create a file contains one SMILES a line.
 And pass the file path to the `-t` or `--train` option.
```
  -t TRAIN, --train TRAIN
                        Training file name (default: train)
  -v VOCAB, --vocab VOCAB
                        Vocab file name (default: vocab)
```

### Evaluation

To start evaluation, use `python reconstruct_eval.py`, and following arguments
```
  -t TRAIN, --train TRAIN
                        Training file name (default: test)
  -v VOCAB, --vocab VOCAB
                        Vocab file name (default: vocab)
  -m MODEL_PATH, --model MODEL_PATH
                        Pre-trained model to be loaded for evalutaion. If not
                        specified, would use pre-trained model from model zoo
                        (default: None)
  -w HIDDEN_SIZE, --hidden HIDDEN_SIZE
                        Hidden size of representation vector, should be
                        consistent with pre-trained model (default: 450)
  -l LATENT_SIZE, --latent LATENT_SIZE
                        Latent Size of node(atom) features and edge(atom)
                        features, should be consistent with pre-trained model
                        (default: 56)
  -d DEPTH, --depth DEPTH
                        Depth of message passing hops, should be consistent
                        with pre-trained model (default: 3)
```

And it would print out the success rate of reconstructing the same molecules.

### Pre-trained models

Below gives the statistics of pre-trained `JTNN_ZINC` model. 

| Pre-trained model  | % Reconstruction Accuracy
| ------------------ | -------
| `JTNN_ZINC`        |  73.7             
