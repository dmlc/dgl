# Temporal Graph Neural Network (TGN)

## DGL Implementation of tgn paper.

This DGL examples implements the GNN mode proposed in the paper [TemporalGraphNeuralNetwork](https://arxiv.org/abs/2006.10637.pdf)

## TGN implementor

This example was implemented by [Ericcsr](https://github.com/Ericcsr) during his SDE internship at the AWS Shanghai AI Lab.

## Graph Dataset

Jodie Wikipedia Temporal dataset. Dataset summary:

- Num Nodes: 9227
- Num Edges: 157, 474
- Num Edge Features: 172
- Edge Feature type: LIWC
- Time Span: 30 days
- Chronological Split: Train: 70% Valid: 15% Test: 15%

Jodie Reddit Temporal dataset. Dataset summary:

- Num Nodes: 11,000
- Num Edges: 672, 447
- Num Edge Features: 172
- Edge Feature type: LIWC
- Time Span: 30 days
- Chronological Split: Train: 70% Valid: 15% Test: 15%

## How to run example files

In tgn folder, run

**please use `train.py`**

```python
python train.py --dataset wikipedia
```

If you want to run in fast mode:

```python
python train.py --dataset wikipedia --fast_mode
```

If you want to run in simple mode:

```python
python train.py --dataset wikipedia --simple_mode
```

If you want to change memory updating module:

```python
python train.py --dataset wikipedia --memory_updater [rnn/gru]
```

If you want to use TGAT:

```python
python train.py --dataset wikipedia --not_use_memory --k_hop 2
```

## Performance

#### Without New Node in test set

| Models/Datasets | Wikipedia          | Reddit           |
| --------------- | ------------------ | ---------------- |
| TGN simple mode | AP: 98.5 AUC: 98.9 | AP: N/A AUC: N/A |
| TGN fast mode   | AP: 98.2 AUC: 98.6 | AP: N/A AUC: N/A |
| TGN             | AP: 98.9 AUC: 98.5 | AP: N/A AUC: N/A |

#### With New Node in test set

| Models/Datasets | Wikipedia           | Reddit           |
| --------------- | ------------------- | ---------------- |
| TGN simple mode | AP: 98.2  AUC: 98.6 | AP: N/A AUC: N/A |
| TGN fast mode   | AP: 98.0  AUC: 98.4 | AP: N/A AUC: N/A |
| TGN             | AP: 98.2  AUC: 98.1 | AP: N/A AUC: N/A |

## Training Speed / Batch
Intel E5 2cores, Tesla K80, Wikipedia Dataset

| Models/Datasets | Wikipedia | Reddit   |
| --------------- | --------- | -------- |
| TGN simple mode | 0.3s      | N/A      |
| TGN fast mode   | 0.28s     | N/A      |
| TGN             | 1.3s      | N/A      |

### Details explained

**What is Simple Mode**

Simple Temporal Sampler just choose the edges that happen before the current timestamp and build the subgraph of the corresponding nodes. 
And then the simple sampler uses the static graph neighborhood sampling methods.

**What is Fast Mode**

Normally temporal encoding needs each node to use incoming time frame as current time which might lead to two nodes have multiple interactions within the same batch need to maintain multiple embedding features which slow down the batching process to avoid feature duplication, fast mode enables fast batching since it uses last memory update time in the last batch as temporal encoding benchmark for each node. Also within each batch, all interaction between two nodes are predicted using the same set of embedding feature

**What is New Node test**

To test the model has the ability to predict link between unseen nodes based on neighboring information of seen nodes. This model deliberately select 10 % of node in test graph and mask them out during the training.

**Why the attention module is not exactly same as TGN original paper**

Attention module used in this model is adapted from DGL GATConv, considering edge feature and time encoding. It is more memory efficient and faster to compute then the attention module proposed in the paper, meanwhile, according to our test, the accuracy of our module compared with the one in paper is the same.


