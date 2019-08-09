# Graph Convolutional Networks for Molecule Property Prediction

## Dependencies
- PyTorch 1.1.0
    - Check the [official website](https://pytorch.org/) for installation guide
- pandas 0.24.2
    - Install with either `conda install pandas` or `pip install pandas`
- RDKit 2019.03.2
    - We recommend installation with `conda install -c conda-forge rdkit`. For other installation recipes, see the 
    [official documentation](https://www.rdkit.org/docs/Install.html).

## Modeling

## Dataset

**Tox21**. The ["Toxicology in the 21st Century" (Tox21)](https://tripod.nih.gov/tox21/challenge/) initiative created a 
public database measuring toxicity of compounds, which has been used in the 2014 Tox21 Data Challenge. The dataset 
contains qualitative toxicity measurements for 8014 compounds on 12 different targets, including nuclear receptors and 
stress response pathways. 

MoleculeNet [1] randomly splits the dataset into training, validation and test set with a 80/10/10 ratio. The authors
of the MoleculeNet performed a study on the dataset with multiple baselines, among which graph convolution models
perform the best on the test set in terms of ROC-AUC score.

## Usage

## Performance
