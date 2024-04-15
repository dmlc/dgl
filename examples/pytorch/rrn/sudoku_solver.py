import os
import urllib.request

import numpy as np
import torch
from sudoku import SudokuNN
from sudoku_data import _basic_sudoku_graph


def solve_sudoku(puzzle):
    """
    Solve sudoku puzzle using RRN.
    :param puzzle: an array-like data with shape [9, 9], blank positions are filled with 0
    :return: a [9, 9] shaped numpy array
    """
    puzzle = np.array(puzzle, dtype=int).reshape([-1])
    model_path = "ckpt"
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    model_filename = os.path.join(model_path, "rrn-sudoku.pkl")
    if not os.path.exists(model_filename):
        print("Downloading model...")
        url = "https://data.dgl.ai/models/rrn-sudoku.pkl"
        urllib.request.urlretrieve(url, model_filename)

    model = SudokuNN(num_steps=64, edge_drop=0.0)
    model.load_state_dict(torch.load(model_filename, map_location="cpu"))
    model.eval()

    g = _basic_sudoku_graph()
    sudoku_indices = np.arange(0, 81)
    rows = sudoku_indices // 9
    cols = sudoku_indices % 9

    g.ndata["row"] = torch.tensor(rows, dtype=torch.long)
    g.ndata["col"] = torch.tensor(cols, dtype=torch.long)
    g.ndata["q"] = torch.tensor(puzzle, dtype=torch.long)
    g.ndata["a"] = torch.tensor(puzzle, dtype=torch.long)

    pred, _ = model(g, False)
    pred = pred.cpu().data.numpy().reshape([9, 9])
    return pred


if __name__ == "__main__":
    q = [
        [9, 7, 0, 4, 0, 2, 0, 5, 3],
        [0, 4, 6, 0, 9, 0, 0, 0, 0],
        [0, 0, 8, 6, 0, 1, 4, 0, 7],
        [0, 0, 0, 0, 0, 3, 5, 0, 0],
        [7, 6, 0, 0, 0, 0, 0, 8, 2],
        [0, 0, 2, 8, 0, 0, 0, 0, 0],
        [6, 0, 5, 1, 0, 7, 2, 0, 0],
        [0, 0, 0, 0, 6, 0, 7, 4, 0],
        [4, 3, 0, 2, 0, 9, 0, 6, 1],
    ]

    answer = solve_sudoku(q)
    print(answer)
