# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import subprocess
import os
from pathlib import Path
import numpy as np
import tempfile

base_path = Path("~/regression/dgl/")

class PartitionBenchmark:

    params = [['pytorch'], ['livejournal']]
    param_names = ['backend', 'dataset']
    timeout = 600

    def __init__(self):
        self.std_log = {}

    def setup(self, backend, dataset):
        key_name = "{}_{}".format(backend, dataset)
        if key_name in self.std_log:
            return
        bench_path = base_path / "tests/regression/benchmarks/partition.py"
        bashCommand = "/opt/conda/envs/{}-ci/bin/python {} --dataset {}".format(
            backend, bench_path.expanduser(), dataset)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE,env=dict(os.environ, DGLBACKEND=backend))
        output, error = process.communicate()
        print(str(error))
        self.std_log[key_name] = str(output)


    def track_partition_time(self, backend, dataset):
        key_name = "{}_{}".format(backend, dataset)
        lines = self.std_log[key_name].split("\\n")

        time_list = []
        for line in lines:
            # print(line)
            if 'Time:' in line:
                time_str = line.strip().split(' ')[1]
                time = float(time_str)
                time_list.append(time)
        return np.array(time_list).mean()


PartitionBenchmark.track_partition_time.unit = 's'

