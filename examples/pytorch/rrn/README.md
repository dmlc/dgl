# Recurrent Relational Network (RRN)

* Paper link: https://arxiv.org/abs/1711.08028
* Author's code repo: https://github.com/rasmusbergpalm/recurrent-relational-networks.git

## Dependencies

* PyTorch 1.0+
* DGL 0.3+

## Codes

The folder contains a DGL implementation of Recurrent Relational Network, and its
application on sudoku solving.

## Results

Run the following
```
python3 train_sudoku.py --output_dir out/ --do_train --do_eval
```
Test accuracy (puzzle-level): 96.08% (paper: 96.6%)
