# Recurrent Relational Network (RRN)

* Paper link: https://arxiv.org/abs/1711.08028
* Author's code repo: https://github.com/rasmusbergpalm/recurrent-relational-networks

## Dependencies

* PyTorch 1.0+
* DGL 0.5+

## Codes

The folder contains a DGL implementation of Recurrent Relational Network, and its
application on sudoku solving.

## Usage

- To train the RRN for sudoku, run the following
```
python3 train_sudoku.py --output_dir out/ --do_train
```

- Test with specified aggregation steps:
```
python3 train_sudoku.py --output_dir out/ --do_eval --steps 64
```

  Test accuracy (puzzle-level): 

|       | 32 steps | 64 steps |
| ----- | :------: | :------: |
| Paper | 94.1     | 96.6     |
| DGL   | 95.3     | 98.9     |


- To use the trained model for solving sudoku, follow the example bellow:

```python
from sudoku_solver import solve_sudoku

q = [[9, 7, 0, 4, 0, 2, 0, 5, 3],
     [0, 4, 6, 0, 9, 0, 0, 0, 0],
     [0, 0, 8, 6, 0, 1, 4, 0, 7],
     [0, 0, 0, 0, 0, 3, 5, 0, 0],
     [7, 6, 0, 0, 0, 0, 0, 8, 2],
     [0, 0, 2, 8, 0, 0, 0, 0, 0],
     [6, 0, 5, 1, 0, 7, 2, 0, 0],
     [0, 0, 0, 0, 6, 0, 7, 4, 0],
     [4, 3, 0, 2, 0, 9, 0, 6, 1]
    ]

answer = solve_sudoku(q)
print(answer)
'''
[[9 7 1 4 8 2 6 5 3]
 [3 4 6 7 9 5 1 2 8]
 [2 5 8 6 3 1 4 9 7]
 [8 1 4 9 2 3 5 7 6]
 [7 6 3 5 1 4 9 8 2]
 [5 9 2 8 7 6 3 1 4]
 [6 8 5 1 4 7 2 3 9]
 [1 2 9 3 6 8 7 4 5]
 [4 3 7 2 5 9 8 6 1]]
'''
```
