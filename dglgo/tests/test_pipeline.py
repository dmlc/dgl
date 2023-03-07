import subprocess
from pathlib import Path
from typing import NamedTuple

import pytest

# class DatasetSpec:

dataset_spec = {"cora": {"timeout": 30}}


class ExperimentSpec(NamedTuple):
    pipeline: str
    dataset: str
    model: str
    timeout: int
    extra_cfg: dict = {}


exps = [
    ExperimentSpec(
        pipeline="nodepred", dataset="cora", model="sage", timeout=0.5
    )
]


@pytest.mark.parametrize("spec", exps)
def test_train(spec):
    cfg_path = "/tmp/test.yaml"
    run = subprocess.run(
        [
            "dgl",
            "config",
            spec.pipeline,
            "--data",
            spec.dataset,
            "--model",
            spec.model,
            "--cfg",
            cfg_path,
        ],
        timeout=spec.timeout,
        capture_output=True,
    )
    assert (
        run.stderr is None or len(run.stderr) == 0
    ), "Found error message: {}".format(run.stderr)
    output = run.stdout.decode("utf-8")
    print(output)

    run = subprocess.run(
        ["dgl", "train", "--cfg", cfg_path],
        timeout=spec.timeout,
        capture_output=True,
    )
    assert (
        run.stderr is None or len(run.stderr) == 0
    ), "Found error message: {}".format(run.stderr)
    output = run.stdout.decode("utf-8")
    print(output)


TEST_RECIPE_FOLDER = "my_recipes"


@pytest.fixture
def setup_recipe_folder():
    run = subprocess.run(
        ["dgl", "recipe", "copy", "--dir", TEST_RECIPE_FOLDER],
        timeout=15,
        capture_output=True,
    )


@pytest.mark.parametrize(
    "file", [str(f) for f in Path(TEST_RECIPE_FOLDER).glob("*.yaml")]
)
def test_recipe(file, setup_recipe_folder):
    print("DGL enter train {}".format(file))
    try:
        run = subprocess.run(
            ["dgl", "train", "--cfg", file], timeout=5, capture_output=True
        )
        sh_stdout, sh_stderr = run.stdout, run.stderr
    except subprocess.TimeoutExpired as e:
        sh_stdout = e.stdout
        sh_stderr = e.stderr
    if sh_stderr is not None and len(sh_stderr) != 0:
        error_str = sh_stderr.decode("utf-8")
        lines = error_str.split("\n")
        for line in lines:
            line = line.strip()
            if (
                line.startswith("WARNING")
                or line.startswith("Aborted")
                or line.startswith("0%")
            ):
                continue
            else:
                assert len(line) == 0, error_str
    print("{} stdout: {}".format(file, sh_stdout))
    print("{} stderr: {}".format(file, sh_stderr))


# test_recipe( , None)
