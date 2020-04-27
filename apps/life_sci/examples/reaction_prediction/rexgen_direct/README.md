# A graph-convolutional neural network model for the prediction of chemical reactivity

- [paper in Chemical Science](https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc04228d#!divAbstract)
- [authors' code](https://github.com/connorcoley/rexgen_direct)

An earlier version of the work was published in NeurIPS 2017 as 
["Predicting Organic Reaction Outcomes with Weisfeiler-Lehman Network"](https://arxiv.org/abs/1709.04555) with some 
slight difference in modeling.

## Dataset

The example by default works with reactions from USPTO (United States Patent and Trademark) granted patents, 
collected by Lowe [1]. After removing duplicates and erroneous reactions, the authors obtain a set of 480K reactions. 
The dataset is divided into 400K, 40K, and 40K for training, validation and test.

## Reaction Center Prediction

### Modeling

Reaction centers refer to the pairs of atoms that lose/form a bond in the reactions. A Graph neural network 
(Weisfeiler-Lehman Network in this case) is trained to update the representations of all atoms. Then we combine 
pairs of atom representations to predict the likelihood for the corresponding atoms to form/lose a bond.

For evaluation, we select pairs of atoms with top-k scores for each reaction and compute the proportion of reactions 
whose reaction centers have all been selected.

### Training with Default Options

We use GPU whenever possible. To train the model with default options, simply do 

```bash
python find_reaction_center.py
```

Once the training process starts, the progress will be printed out in the terminal as follows:

```bash
Epoch 1/50, iter 8150/20452 | time/minibatch 0.0260 | loss 8.4788 | grad norm 12.9927
Epoch 1/50, iter 8200/20452 | time/minibatch 0.0260 | loss 8.6722 | grad norm 14.0833
```

After an epoch of training is completed, we evaluate the model on the validation set and 
print the evaluation results as follows:

```bash
Epoch 4/50, validation | acc@10 0.8213 | acc@20 0.9016 |
```

By default, we store the model per 10000 iterations in `center_results`.

**Speedup**: For an epoch of training, our implementation takes about 5095s for the first epoch  while the authors' 
implementation takes about 11657s, which is roughly a speedup by 2.3x.

For model evaluation, we can choose whether to exclude reactants not contributing heavy atoms to the product 
(e.g. reagents and solvents) in top-k atom pair selection, which will make the task easier. 
For the easier evaluation, do

```bash
python find_reaction_center.py --easy
```

A summary of the model performance is as follows:

| Item            | Top 6 accuracy | Top 8 accuracy | Top 10 accuracy |
| --------------- | -------------- | -------------- | --------------- |
| Paper           | 89.8           | 92.0           | 93.3            |
| Hard evaluation | 88.8           | 91.6           | 92.9            |
| Easy evaluation | 91.0           | 93.7           | 94.9            |

### Pre-trained Model

We provide a pre-trained model so users do not need to train from scratch. To evaluate the pre-trained model, simply do

```bash
python find_reaction_center.py -p
```

### Adapting to a new dataset.

New datasets should be processed such that each line corresponds to the SMILES for a reaction like below:

```bash
[CH3:14][NH2:15].[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[Cl:13].[OH2:16]>>[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[NH:15][CH3:14]
```

The reactants are placed before `>>` and the product is placed after `>>`. The reactants are separated by `.`. 
In addition, atom mapping information is provided.

You can then train a model on new datasets with 

```bash
python find_reaction_center.py --train-path X --val-path Y --test-path Z
```

where `X`, `Y`, `Z` are paths to the new training/validation/test set as described above.

## References

[1] D. M.Lowe, Patent reaction extraction: downloads, 
https://bitbucket.org/dan2097/patent-reaction-extraction/downloads, 2014.
