# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import subprocess
import os
from pathlib import Path

base_path = "~/regression/dgl/"

class GCNBenchmark:

    def setup(self):
        print(os.getcwd())
        bashCommand = "python examples/pytorch/gcn/train.py --dataset cora --gpu -1"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        self.output = output
    
    def track_gcn_time(self):
        print(self.output)
        return 1
    
    def track_gcn_accuracy(self):
        print(self.output)
        print(222)
        return 2


GCNBenchmark.track_gcn_time.unit = 's'
GCNBenchmark.track_gcn_accuracy.unit = '%'