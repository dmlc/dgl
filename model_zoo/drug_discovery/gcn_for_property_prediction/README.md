# Graph Convolutional Networks for Molecule Property Prediction

We consider this model to be the first step stone for drug discovery with DGL. If you have used DeepChem before, you
should find our model very similar to 
[DeepChem's tutorial on graph convolution](https://deepchem.io/docs/notebooks/graph_convolutional_networks_for_tox21.html).

## Dependencies
- PyTorch 1.1.0
    - Check the [official website](https://pytorch.org/) for installation guide
- pandas 0.24.2
    - Install with either `conda install pandas` or `pip install pandas`
- RDKit 2019.03.2
    - We recommend installation with `conda install -c conda-forge rdkit`. For other installation recipes, see the 
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

## Usage

`python main.py`

To use gpu

`python main.py -c`

To skip training and use a pre-trained model for test

`python main.py -p`

Other options include:
- Set random seed to be x with `-s x`
- Set number of epochs for training to be x with `-ne x`
- Set training batch size to be x with `-b x`
- Set learning rate to be x with `-lr x`
- Set dropout rate to be x with `-d x`
- Set path for model checkpoint to be x with `-cp x`

## Performance

## References
