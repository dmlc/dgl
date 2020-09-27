#-*- coding:utf-8 -*-

# Author:james Zhang
# Datetime:20-3-7 下午6:13
# Project: RED
"""
    utilities file for Pyorch models

"""

from functools import wraps
import traceback
from _thread import start_new_thread
import torch.multiprocessing as mp


class early_stopper(object):
    """
    patience: Wait how many rounds that their loss is larger than the best

    verbose: Boolean, if print out log information

    delta: tolarance range of loss, e.g. if delta is 0.01, the best value must be smaller than pre best - delta
    """
    def __init__(self, patience=10, verbose=False, delta=0):

        self.patience = patience
        self.verbose = verbose
        self.delta = delta

        self.best_value = None
        self.is_earlystop = False
        self.count = 0
        self.val_preds = []
        self.val_logits = []

    def earlystop(self, loss, preds, logits):

        value = -loss

        if self.best_value is None:
            self.best_value = value
            self.val_preds = preds
            self.val_logits = logits
        elif value < self.best_value + self.delta:
            self.count += 1
            if self.verbose:
                print('EarlyStoper count: {:02d}'.format(self.count))
            if self.count >= self.patience:
                self.is_earlystop = True
        else:
            self.best_value = value
            self.val_preds = preds
            self.val_logits = logits
            self.count = 0


# According to https://github.com/pytorch/pytorch/issues/17199, this decorator
# is necessary to make fork() and openmp work together.
def thread_wrapped_func(func):
    """
    Wraps a process entry point to make it work with OpenMP.
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = mp.Queue()
        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)
    return decorated_function