# A graph-convolutional neural network model for the prediction of chemical reactivity

- [paper in Chemical Science](https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc04228d#!divAbstract)
- [authors' code](https://github.com/connorcoley/rexgen_direct)

An earlier version of the work was published in NeurIPS 2017 as 
["Predicting Organic Reaction Outcomes with Weisfeiler-Lehman Network"](https://arxiv.org/abs/1709.04555) with some 
slight difference in modeling.

This work proposes a template-free approach for reaction prediction with 2 stages: 
1) Identify reaction center (pairs of atoms that will lose a bond or form a bond)
2) Enumerate the possible combinations of bond changes

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
python find_reaction_center_train.py
```

Once the training process starts, the progress will be printed in the terminal as follows:

```bash
Epoch 1/50, iter 8150/20452 | loss 8.4788 | grad norm 12.9927
Epoch 1/50, iter 8200/20452 | loss 8.6722 | grad norm 14.0833
```

Everytime the learning rate is decayed (specified as `'decay_every'` in `configure.py`), we save a checkpoint of 
the model and evaluate the model on the validation set. The evaluation result is as follows, where `acc@k` means 
top-k accuracy:

```bash
total iter 40000, (epoch 2/35, iter 2443/2557) | acc@12 0.9278 | acc@16 0.9419 | acc@20 0.9496 | acc@40 0.9596 | acc@80 0.9596 |
```

All model check points and evaluation results can be found under `center_results`. `model_x.pkl` stores a model 
checkpoint after training for `x` iterations, or equivalently `x * batch_size` samples. `val_eval.txt` stores all 
evaluation results on the validation set as follows:

```bash
total iter 10000, (epoch 1/35, iter 1250/2557) | acc@12 0.8907 | acc@16 0.9104 | acc@20 0.9227 | acc@40 0.9394 | acc@80 0.9394 |
total iter 20000, (epoch 1/35, iter 2500/2557) | acc@12 0.9125 | acc@16 0.9287 | acc@20 0.9387 | acc@40 0.9512 | acc@80 0.9512 |
total iter 30000, (epoch 2/35, iter 1193/2557) | acc@12 0.9218 | acc@16 0.9370 | acc@20 0.9458 | acc@40 0.9563 | acc@80 0.9563 |
...
```

You may want to terminate the training process when the validation performance no longer improves for some time.

### Multi-GPU Training

By default we use one GPU only. We also allow multi-gpu training. To use GPUs with ids `id1,id2,...`, do

```bash
python find_reaction_center_train.py --gpus id1,id2,...
```

A summary of the training speedup with the DGL implementation is presented below.

| Item | Training time (s/epoch) | Speedup | 
| ---- | ----------------------- | ------- |
| Authors' implementation | 11657 | 1x |
| DGL with 1 gpu  | 858 | 13.6x |
| DGL with 2 gpus | 443 | 26.3x |
| DGL with 4 gpus | 243 | 48.0x |
| DGL with 8 gpus | 134 | 87.0x |

### Evaluation

```bash
python find_reaction_center_eval.py --model-path X
```

For example, you can evaluate the model trained for 10000 iterations by setting `X` to be 
`center_results/model_10000.pkl`. The evaluation results will be stored at `center_results/test_eval.txt`.

For model evaluation, we can choose whether to exclude reactants not contributing heavy atoms to the product 
(e.g. reagents and solvents) in top-k atom pair selection, which will make the task easier. 
For the easier evaluation, do

```bash
python find_reaction_center_eval.py --easy
```

A summary of the model performance of various settings is as follows:

| Item            | Top 6 accuracy | Top 8 accuracy | Top 10 accuracy |
| --------------- | -------------- | -------------- | --------------- |
| Paper           | 89.8           | 92.0           | 93.3            |
| Hard evaluation from authors' code | 87.7           | 90.6           |  92.1           |
| Easy evaluation from authors' code | 90.0           | 92.8           |  94.2           |
| Hard evaluation | 88.9           | 91.7           | 93.1            |
| Easy evaluation | 91.2           | 93.8           | 95.0            |
| Hard evaluation for model trained on 8 gpus | 88.1 | 91.0 | 92.5 |
| Easy evaluation for model trained on 8 gpus | 90.3 | 93.3 | 94.6 |

1. We are able to match the results reported from authors' code for both single-gpu and multi-gpu training
2. While multi-gpu training provides a great speedup, the performance with the default hyperparameters drops slightly.

### Data Pre-processing with Multi-Processing

By default we use 32 processes for data pre-processing. If you encounter an error with 
`BrokenPipeError: [Errno 32] Broken pipe`, you can specify a smaller number of processes with 
```bash
python find_reaction_center_train.py -np X
```
```bash
python find_reaction_center_eval.py -np X
```
where `X` is the number of processes that you would like to use.

### Pre-trained Model

We provide a pre-trained model so users do not need to train from scratch. To evaluate the pre-trained model, simply do

```bash
python find_reaction_center_eval.py
```

### Adapting to a New Dataset

New datasets should be processed such that each line corresponds to the SMILES for a reaction like below:

```bash
[CH3:14][NH2:15].[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[Cl:13].[OH2:16]>>[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[NH:15][CH3:14]
```

The reactants are placed before `>>` and the product is placed after `>>`. The reactants are separated by `.`. 
In addition, atom mapping information is provided.

You can then train a model on new datasets with 

```bash
python find_reaction_center_train.py --train-path X --val-path Y
```

where `X`, `Y` are paths to the new training/validation as described above.

For evaluation,

```bash
python find_reaction_center_eval.py --eval-path Z
```

where `Z` is the path to the new test set as described above.

## Candidate Ranking

For candidate ranking, we assume that a model has been trained for reaction center prediction first. 
By default, we exclude reactants not contributing heavy atoms to the product in selecting candidate bonds 
for a consistent setting as in the paper.

### Additional Dependency

In addition to RDKit, MolVS is an alternative for comparing whether two molecules are the same after sanitization.

- [molvs](https://molvs.readthedocs.io/en/latest/)

## References

[1] D. M.Lowe, Patent reaction extraction: downloads, 
https://bitbucket.org/dan2097/patent-reaction-extraction/downloads, 2014.
