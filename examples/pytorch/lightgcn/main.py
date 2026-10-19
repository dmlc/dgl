from dataset import Dataset
import torch
import numpy as np
from config import args
from procedure import *


if __name__ == "__main__":
    dataset = Dataset(args.dataset)
    train_lightgcn(dataset)