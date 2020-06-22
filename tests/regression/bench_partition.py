# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import subprocess
import os
from pathlib import Path
import numpy as np
import tempfile

base_path = Path("~/regression/dgl/")

class Benchmark:

    params = [['pytorch'], ['livejournal']]
    param_names = ['backend', 'dataset']
    timeout = 600

    def __init__(self):
        self.std_log = {}

    def setup(self, backend, dataset):
        key_name = "{}_{}".format(backend, dataset)
        if key_name in self.std_log:
            return
        bench_path = base_path / "tests/regression/benchmarks/benchmark.py"
        bashCommand = "/opt/conda/envs/{}-ci/bin/python {} --dataset {}".format(
            backend, bench_path.expanduser(), dataset)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE,
                                   env=dict(os.environ, DGLBACKEND=backend))
        output, error = process.communicate()
        print(str(error))
        self.std_log[key_name] = str(output)


    def track_all_time(self, backend, dataset):
        key_name = "{}_{}".format(backend, dataset)
        lines = self.std_log[key_name].split("\\n")

        full_graph_times = []
        mini_batch_times = []
        for line in lines:
            if 'full_graph' in line:
                strs = line.strip().split(' ')
                name = strs[1]
                time_str = strs[2]
                time = float(time_str)
                full_graph_times.append(time)
            if 'mini_batch' in line:
                strs = line.strip().split(' ')
                name = strs[1]
                time_str = strs[2]
                time = float(time_str)
                mini_batch_times.append(time)
        return np.array(full_graph_times)


PartitionBenchmark.track_all_time.unit = 's'

