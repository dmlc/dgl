# A graph-convolutional neural network model for the prediction of chemical reactivity

- [paper in Chemical Science](https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc04228d#!divAbstract)
- [authors' code](https://github.com/connorcoley/rexgen_direct)

An earlier version of the work was published in NeurIPS 2017 as 
["Predicting Organic Reaction Outcomes with Weisfeiler-Lehman Network"](https://arxiv.org/abs/1709.04555) with some 
slight difference in modeling.

This work proposes a template-free approach for reaction prediction with 2 stages: 
1) Identify reaction center (pairs of atoms that will lose a bond or form a bond)
2) Enumerate the possible combinations of bond changes and rank the corresponding candidate products

We provide a jupyter notebook for walking through a demonstration with our pre-trained models. You can 
download it with `wget https://data.dgl.ai/dgllife/reaction_prediction_pretrained.ipynb` and you need to put it 
in this directory. Below we visualize a reaction prediction by the model:

![](https://data.dgl.ai/dgllife/wln_reaction.png)

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

Everytime the learning rate is decayed (specified as `'decay_every'` in `configure.py`'s `reaction_center_config`), we save a checkpoint of 
the model and evaluate the model on the validation set. The evaluation result is formatted as follows, where `total samples x` means 
the we have trained the model on `x` samples and `acc@k` means top-k accuracy:

```bash
total samples 800000, (epoch 2/35, iter 2443/2557) | acc@12 0.9278 | acc@16 0.9419 | acc@20 0.9496 | acc@40 0.9596 | acc@80 0.9596 |
```

All model check points and evaluation results can be found under `center_results`. `model_x.pkl` stores a model 
checkpoint after seeing `x` training samples. `val_eval.txt` stores all 
evaluation results on the validation set.

You may want to terminate the training process when the validation performance no longer improves for some time.

### Multi-GPU Training

By default we use one GPU only. We also allow multi-gpu training. To use GPUs with ids `id1,id2,...`, do

```bash
python find_reaction_center_train.py --gpus id1,id2,...
```

A summary of the training speedup with the DGL implementation is presented below.

| Item                    | Training time (s/epoch) | Speedup | 
| ----------------------- | ----------------------- | ------- |
| Authors' implementation | 11657                   | 1x      |
| DGL with 1 gpu          | 858                     | 13.6x   |
| DGL with 2 gpus         | 443                     | 26.3x   |
| DGL with 4 gpus         | 243                     | 48.0x   |
| DGL with 8 gpus         | 134                     | 87.0x   | 

### Evaluation

```bash
python find_reaction_center_eval.py --model-path X
```

For example, you can evaluate the model trained for 800000 samples by setting `X` to be 
`center_results/model_800000.pkl`. The evaluation results will be stored at `center_results/test_eval.txt`.

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

### Additional Dependency

In addition to RDKit, MolVS is an alternative for comparing whether two molecules are the same after sanitization.

- [molvs](https://molvs.readthedocs.io/en/latest/)

### Modeling

For candidate ranking, we assume that a model has been trained for reaction center prediction first. 
The pipeline for predicting candidate products given a reaction proceeds as follows:
1. Select top-k bond changes for atom pairs in the reactants, ranked by the model for reaction center prediction. 
By default, we use k=80 and exclude reactants not contributing heavy atoms to the ground truth product in 
selecting top-k bond changes as in the paper.
2. Filter out candidate bond changes for bonds that are already in the reactants
3. Enumerate possible combinations of atom pairs with up to C pairs, which reflects the number of bond changes 
(losing or forming a bond) in reactions. A statistical analysis in USPTO suggests that setting it to 5 is enough.
4. Filter out invalid combinations where 1) atoms in candidate bond changes are not connected or 2) an atom pair is 
predicted to have different types of bond changes 
(e.g. two atoms are predicted simultaneously to form a single and double bond) or 3) valence constraints are violated.
5. Apply the candidate bond changes for each valid combination and get the corresponding candidate products.
6. Construct molecular graphs for the reactants and candidate products, featurize their atoms and bonds.
7. Apply a Weisfeiler-Lehman Network to the molecular graphs for reactants and candidate products and score them

### Training with Default Options

We use GPU whenever possible. To train the model with default options, simply do 

```bash
python candidate_ranking_train.py -cmp X
```

where `X` is the path to a trained model for reaction center prediction. You can use our 
pre-trained model by not specifying `-cmp`.

Once the training process starts, the progress will be printed in the terminal as follows:

```bash
Epoch 6/6, iter 16439/20061 | time 1.1124 | accuracy 0.8500 | grad norm 5.3218
Epoch 6/6, iter 16440/20061 | time 1.1124 | accuracy 0.9500 | grad norm 2.1163
```

Everytime the learning rate is decayed (specified as `'decay_every'` in `configure.py`'s `candidate_ranking_config`), 
we save a checkpoint of the model and evaluate the model on the validation set. The evaluation result is formatted 
as follows, where `total samples x` means that we have trained the model for `x` samples, `acc@k` means top-k accuracy, 
`gfound` means the proportion of reactions where the ground truth product can be recovered by the ground truth bond changes. 
We perform the evaluation based on RDKit-sanitized molecule equivalence (marked with `[strict]`) and MOLVS-sanitized 
molecule equivalence (marked with `[molvs]`).

```bash
total samples 100000, (epoch 1/20, iter 5000/20061) 
[strict] acc@1: 0.7732 acc@2: 0.8466 acc@3: 0.8763 acc@5: 0.8987 gfound 0.9864
[molvs] acc@1: 0.7779 acc@2: 0.8523 acc@3: 0.8826 acc@5: 0.9057 gfound 0.9953
```

All model check points and evaluation results can be found under `candidate_results`. `model_x.pkl` stores a model 
checkpoint after seeing `x` training samples in total. `val_eval.txt` stores all 
evaluation results on the validation set.

You may want to terminate the training process when the validation performance no longer improves for some time.

### Evaluation

```bash
python candidate_ranking_eval.py --model-path X -cmp Y
```

where `X` is the path to a trained model for candidate ranking and `Y` is the path to a trained model 
for reaction center prediction. For example, you can evaluate the model trained for 800000 samples by setting `X` to be 
`candidate_results/model_800000.pkl`. The evaluation results will be stored at `candidate_results/test_eval.txt`. As 
in training, you can use our pre-trained model by not specifying `-cmp`.

A summary of the model performance of various settings is as follows:

| Item                       | Top 1 accuracy | Top 2 accuracy | Top 3 accuracy | Top 5 accuracy |
| -------------------------- | -------------- | -------------- | -------------- | -------------- |
| Authors' strict evaluation | 85.6           | 90.5           | 92.8           | 93.4           |
| DGL's strict evaluation    | 85.6           | 90.0           | 91.7           | 92.9           |
| Authors' molvs evaluation  | 86.2           | 91.2           | 92.8           | 94.2           |
| DGL's molvs evaluation     | 86.1           | 90.6           | 92.4           | 93.6           |

### Pre-trained Model

We provide a pre-trained model so users do not need to train from scratch. To evaluate the pre-trained model, 
simply do

```bash
python candidate_ranking_eval.py
```

### Adapting to a New Dataset

You can train a model on new datasets with

```bash
python candidate_ranking_train.py --train-path X --val-path Y
```

where `X`, `Y` are paths to the new training/validation set as described in the `Reaction Center Prediction` section.

For evaluation,

```bash
python candidate_ranking_train.py --eval-path Z
```

where `Z` is the path to the new test set as described in the `Reaction Center Prediction` section.

## References

[1] D. M.Lowe, Patent reaction extraction: downloads, 
https://bitbucket.org/dan2097/patent-reaction-extraction/downloads, 2014.
