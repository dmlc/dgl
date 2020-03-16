# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import subprocess
import os
from pathlib import Path
import numpy as np
import tempfile

base_path = Path("~/regression/dgl/")


class SAGEBenchmark:

    params = [['pytorch'], ['0']]
    param_names = ['backend', 'gpu']
    timeout = 1800

    # def setup_cache(self):
    #     self.tmp_dir = Path(tempfile.mkdtemp())

    def setup(self, backend, gpu):
        log_filename = Path("sage_sampling_{}_{}.log".format(backend, gpu))
        if log_filename.exists():
            return
        run_path = base_path / "examples/{}/graphsage/train_sampling.py".format(backend)
        bashCommand = "/opt/conda/envs/{}-ci/bin/python {} --num-workers=4 --num-epochs=16 --gpu={}".format(
            backend, run_path.expanduser(), gpu)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE,env=dict(os.environ, DGLBACKEND=backend))
        output, error = process.communicate()
        print(str(error))
        log_filename.write_text(str(output))


    def track_sage_time(self, backend):
        log_filename = Path("sage_sampling_{}_{}.log".format(backend, gpu))
        lines = log_filename.read_text().split("\\n")
        time_list = []
        for line in lines:
            if line.startswith('Epoch Time'):
                time_str = line.strip()[15:]
                time_list.append(float(time_str))
        return np.array(time_list).mean()

    def track_sage_accuracy(self, backend):
        log_filename = Path("sage_sampling_{}_{}.log".format(backend, gpu))
        lines = log_filename.read_text().split("\\n")
        test_acc = 0.
        for line in lines:
            if line.startswith('Eval Acc'):
                acc_str = line.strip()[9:]
                test_acc = float(acc_str)
        return test_acc * 100


SAGEBenchmark.track_sage_time.unit = 's'
SAGEBenchmark.track_sage_accuracy.unit = '%'
