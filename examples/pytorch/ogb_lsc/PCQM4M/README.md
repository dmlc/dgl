# Baseline Code for PCQM4M-LSC

The code is ported from the official examples [here](https://github.com/snap-stanford/ogb/tree/master/examples/lsc/pcqm4m). Please refer to the [OGB-LSC paper](https://arxiv.org/abs/2103.09430) for the detailed setting.

## Installation Requirements

```
ogb>=1.3.0
rdkit>=2019.03.1
torch>=1.7.0
```

We recommend installing RDKit with `conda install -c rdkit rdkit==2019.03.1`.

## Commandline Arguments

- `LOG_DIR`: Tensorboard log directory.
- `CHECKPOINT_DIR`: Directory to save the best validation checkpoint. The checkpoint file will be saved at `${CHECKPOINT_DIR}/checkpoint.pt`.
- `TEST_DIR`: Directory path to save the test submission. The test file will be saved at `${TEST_DIR}/y_pred_pcqm4m.npz`.

## Baseline Models

### GIN [1]

```
python main.py --gnn gin --log_dir $LOG_DIR --checkpoint_dir $CHECKPOINT_DIR --save_test_dir $TEST_DIR
```

### GIN-virtual [1,3]

```
python main.py --gnn gin-virtual --log_dir $LOG_DIR --checkpoint_dir $CHECKPOINT_DIR --save_test_dir $TEST_DIR
```

### GCN [2]

```
python main.py --gnn gcn --log_dir $LOG_DIR --checkpoint_dir $CHECKPOINT_DIR --save_test_dir $TEST_DIR
```

### GCN-virtual [2,3]

```
python main.py --gnn gcn-virtual --log_dir $LOG_DIR --checkpoint_dir $CHECKPOINT_DIR --save_test_dir $TEST_DIR
```

## Measuring the Test Inference Time

The code below takes **the raw SMILES strings as input**, uses the saved checkpoint, and performs inference over for all the 377,423 test molecules.

```
python test_inference.py --gnn $GNN --checkpoint_dir $CHECKPOINT_DIR --save_test_dir $TEST_DIR
```

For your model, **the total inference time needs to be less than 12 hours on a single GPU and a CPU**. Ideally, you 
should use the CPU/GPU spec of the organizers, which consists of a single GeForce RTX 2080 GPU and an Intel(R) Xeon(R) 
Gold 6148 CPU @ 2.40GHz. However, the organizers also allow the use of other GPU/CPU specs, as long as the specs are 
clearly reported in the final submission.

## Performance

| Model       | Original Valid MAE | DGL Valid MAE | #Parameters | 
| ----------- | ------------------ | ------------- | ----------- | 
| GIN         | 0.1536             | 0.1536        | 3.8M        | 
| GIN-virtual | 0.1396             | 0.1407        | 6.7M        |
| GCN         | 0.1684             | 0.1683        | 2.0M        |
| GCN-virtual | 0.1510             | 0.1557        | 4.9M        |

## References

[1] Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How powerful are graph neural networks?. ICLR 2019

[2] Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. ICLR 2017

[3] Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. Neural message passing for quantum chemistry. ICML 2017.
