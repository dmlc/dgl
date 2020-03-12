# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import subprocess
import os
from pathlib import Path
import numpy as np
import tempfile

base_path = Path("~/regression/dgl/")


class GCNBenchmark:

    params = [['pytorch'], ['cora', 'pubmed'], ['0', '-1']]
    param_names = ['backend', 'dataset', 'gpu_id']
    timeout = 120

    # def setup_cache(self):
    #     self.tmp_dir = Path(tempfile.mkdtemp())

    def setup(self, backend, dataset, gpu_id):
        log_filename = Path("{}_{}_{}.log".format(backend, dataset, gpu_id))
        if log_filename.exists():
            return
        gcn_path = base_path / "examples/{}/gcn/train.py".format(backend)
        bashCommand = "/opt/conda/envs/{}-ci/bin/python {} --dataset {} --gpu {} --n-epochs 50".format(
            backend, gcn_path.expanduser(), dataset, gpu_id)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE,env=dict(os.environ, DGLBACKEND=backend))
        output, error = process.communicate()
        print(str(error))
        log_filename.write_text(str(output))


    def track_gcn_time(self, backend, dataset, gpu_id):
        log_filename = Path("{}_{}_{}.log".format(backend, dataset, gpu_id))
        lines = log_filename.read_text().split("\\n")
        time_list = []
        for line in lines:
            # print(line)
            if 'Time' in line:
                time_str = line.strip().split('|')[1]
                time = float(time_str.split()[-1])
                time_list.append(time)
        return np.array(time_list)[-10:].mean()

    def track_gcn_accuracy(self, backend, dataset, gpu_id):
        log_filename = Path("{}_{}_{}.log".format(backend, dataset, gpu_id))
        lines = log_filename.read_text().split("\\n")
        test_acc = -1
        for line in lines:
            if 'Test accuracy' in line:
                test_acc = float(line.split()[-1][:-1])
                print(test_acc)
        return test_acc


GCNBenchmark.track_gcn_time.unit = 's'
GCNBenchmark.track_gcn_accuracy.unit = '%'
