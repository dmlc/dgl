import subprocess
from typing import NamedTuple
import pytest
# class DatasetSpec:

dataset_spec = {
    "cora": {"timeout": 30}
}



class ExperimentSpec(NamedTuple):
    pipeline: str
    dataset: str
    model: str
    timeout: int
    extra_cfg: dict = {}

exps = [ExperimentSpec(pipeline="nodepred", dataset="cora", model="sage", timeout=0.5)]

@pytest.mark.parametrize("spec", exps)
def test_train(spec):
    cfg_path = "/tmp/test.yaml"
    run = subprocess.run(["dgl-enter", "config", spec.pipeline, "--data", spec.dataset, "--model", spec.model, "--cfg", cfg_path], timeout=spec.timeout, capture_output=True)
    assert run.stderr is None or len(run.stderr) == 0, "Found error message: {}".format(run.stderr)
    output = run.stdout.decode("utf-8")
    print(output)

    run = subprocess.run(["dgl-enter", "train", "--cfg", cfg_path], timeout=spec.timeout, capture_output=True)
    assert run.stderr is None or len(run.stderr) == 0, "Found error message: {}".format(run.stderr)
    output = run.stdout.decode("utf-8")
    print(output)
