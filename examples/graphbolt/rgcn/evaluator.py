import numpy as np

try:
    import torch
except ImportError:
    torch = None


class IGB_Evaluator:
    def __init__(self, name, num_tasks):
        self.name = name
        self.num_tasks = num_tasks

    def _parse_input(self, input_dict):
        y_true, y_pred = input_dict["y_true"], input_dict["y_pred"]

        if torch and isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if torch and isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()

        if not isinstance(y_true, np.ndarray) or not isinstance(
            y_pred, np.ndarray
        ):
            raise RuntimeError("Arguments must be numpy arrays")

        if y_true.shape != y_pred.shape or y_true.ndim != 2:
            raise RuntimeError("Shape mismatch between y_true and y_pred")

        return y_true, y_pred

    def _eval_acc(self, y_true, y_pred):
        return {"acc": np.mean(np.all(y_true == y_pred, axis=1))}

    def eval(self, input_dict):
        y_true, y_pred = self._parse_input(input_dict)
        return self._eval_acc(y_true, y_pred)
