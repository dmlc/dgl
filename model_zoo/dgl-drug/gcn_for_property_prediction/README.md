# Graph Convolutional Networks for Molecule Property Prediction

We consider this model to be the first step stone for drug discovery with DGL. If you have used DeepChem before, you
should find our model very similar to 
[DeepChem's tutorial on graph convolution](https://deepchem.io/docs/notebooks/graph_convolutional_networks_for_tox21.html).

## Dependencies
- PyTorch 1.2.0
    - Check the [official website](https://pytorch.org/) for installation guide
- pandas 0.24.2
    - Install with either `conda install pandas` or `pip install pandas`
- RDKit 2018.09.3
    - We recommend installation with `conda install -c conda-forge rdkit==2018.09.3`. For other installation recipes, see the 
    [official documentation](https://www.rdkit.org/docs/Install.html).
- requests 2.22.0
    - Install with `pip install requests`

## Dataset

**Tox21**. The ["Toxicology in the 21st Century" (Tox21)](https://tripod.nih.gov/tox21/challenge/) initiative created a 
public database measuring toxicity of compounds, which has been used in the 2014 Tox21 Data Challenge. The dataset 
contains qualitative toxicity measurements for 8014 compounds on 12 different targets, including nuclear receptors and 
stress response pathways. Each target yields a binary prediction problem.

MoleculeNet [1] randomly splits the dataset into training, validation and test set with a 80/10/10 ratio. The authors
of the MoleculeNet performed a study on the dataset with multiple baselines, among which graph convolution models
perform the best on the test set in terms of ROC-AUC score.

## Modeling

In this section we explain the modeling process. Details like dropout and batch normalization are omitted.

### Featurization

We start with molecular graphs. For example below is the molecular graph of `Nc1ccncc1N` visualized by RDKit.

![](https://s3.us-east-2.amazonaws.com/dgl.ai/model_zoo/drug_discovery/gcn_for_property_prediction/rdkit_Nc1ccncc1N.png)

Graph neural networks are able to compute learning based fingerprints (molecule representations) out of local features
and molecular graph topology. In this model we only consider atom features, but it's possible to include more local 
features like bond features. The initial atom features encode information including:
- Atom type, e.g. `C`, `O`
- Atom degree
- Number of implicit Hs on the atom
- Formal charge of the atom
- Number of radical electrons of the atom
- Atom hybridization
- Whether the atom is aromatic
- Number of total Hs on the atom

Note that we do not directly model hydrogen atoms.

### Update Atom Representations with Message Passing/Graph Convolution

A graph convolution layer updates atom representations as follows:

![](https://s3.us-east-2.amazonaws.com/dgl.ai/model_zoo/drug_discovery/gcn_for_property_prediction/message_passing.png)

This operation allows atoms to communicate information with their neighbor atoms along chemical bonds. Therefore, it
is also frequently referred as message passing.

In this model, we use two graph convolution layers. Hence each atom gets to communicate with their two-hop neighbors 
(atoms that can be reached via two chemical bonds).

### From Atom Representations to Molecule Representations

After we update atom representations, each atom now encodes the information of a local subgraph. The molecule
representations can then be computed as follows:

![](https://s3.us-east-2.amazonaws.com/dgl.ai/model_zoo/drug_discovery/gcn_for_property_prediction/fingerprint.png)

In other words, we separately compute a weighted sum of atom representations and an elementwise maximum of atom 
representations. The sum and maximum are then concatenated for a molecule representation.

### Prediction

Finally, we use a multilayer perceptron to make predictions from the computed molecule representations.

## Usage

`python main.py`

To use gpu

`python main.py -c`

To skip training and use a pre-trained model for test

`python main.py -p`

A pre-trained model will be downloaded as `pre_trained.pth`.

Other options include:
- Set random seed to be x with `-s x`
- Set number of epochs for training to be x with `-ne x`
- Set training batch size to be x with `-b x`
- Set learning rate to be x with `-lr x`
- Set dropout rate to be x with `-d x`
- Set path for model checkpoint to be x with `-cp x`

## Performance

| Source           | Averaged ROC-AUC Score |
| ---------------- | ---------------------- |
| MoleculeNet [2]  | 0.829                  |
| [DeepChem example](https://github.com/deepchem/deepchem/blob/master/examples/tox21/tox21_tensorgraph_graph_conv.py) | 0.813                  |
| This model       | 0.827                  |

Note that due to some possible randomness you may get different numbers for DeepChem example and our model. To get
match exact results for this model, please use the pre-trained model as in the usage section.

The training takes less than a minute.

## Next Steps

In this model, we used a pre-built graph convolution layer. If you want to develop your own graph neural networks and 
just get started in DGL, check our [tutorials](https://docs.dgl.ai/tutorials/basics/1_first.html). 

## References

[1] Kipf et al. (2017) Semi-Supervised Classification with Graph Convolutional Networks. 
*The International Conference on Learning Representations (ICLR)*.

[2] Wu et al. (2017) MoleculeNet: a benchmark for molecular machine learning. *Chemical Science* 9, 513-530.
