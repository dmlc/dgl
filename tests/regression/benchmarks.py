# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import subprocess
import os
from pathlib import Path
import numpy as np

base_path = Path("~/regression/dgl/examples/pytorch/gcn/train.py")
    
def track_gcn_time(dataset, gpu_id):
    bashCommand = "python {} --dataset {} --gpu {} --n-epochs 50".format(base_path.expanduser(), dataset, gpu_id)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    lines = self.output.split("\n")
    time_list = []
    for line in lines:
        if 'Time ' in line:
            time_str = lines.strip().split('|')[1]
            time = float(time_str.split()[-1])
            time_list.append(time)
    return np.array(time_list)[-10:].mean()

    
def track_gcn_accuracy(dataset, gpu_id):
    bashCommand = "python {} --dataset {} --gpu {} --n-epochs 50".format(base_path.expanduser(), dataset, gpu_id)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    lines = self.output.split("\n")
    acc_list = []
    for line in lines:
        if 'Test accuracy' in line:
            acc_str = lines.strip().split('|')[3]
            acc = float(time_str.split()[-1])
            acc_list.append(acc)
    return np.array(acc_list)[-10:].mean()


track_gcn_time.unit = 's'
track_gcn_time.params = [['cora', 'pubmed'], ['0', '-1']]
track_gcn_accuracy.unit = '%'
track_gcn_accuracy.params = [['cora', 'pubmed'], ['0', '-1']]