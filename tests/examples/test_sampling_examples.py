import os
import subprocess
import sys
import unittest

EXAMPLE_ROOT = os.path.join(
    os.path.dirname(os.path.relpath(__file__)),
    "..",
    "..",
    "examples",
    "graphbolt",
    "quickstart",
)


def test_node_classification():
    script = os.path.join(EXAMPLE_ROOT, "node_classification.py")
    out = subprocess.run(["python", str(script)], capture_output=True)
    assert (
        out.returncode == 0
    ), f"stdout: {out.stdout.decode('utf-8')}\nstderr: {out.stderr.decode('utf-8')}"
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-5:]) > 0.59


def test_link_prediction():
    script = os.path.join(EXAMPLE_ROOT, "link_prediction.py")
    out = subprocess.run(["python", str(script)], capture_output=True)
    assert (
        out.returncode == 0
    ), f"stdout: {out.stdout.decode('utf-8')}\nstderr: {out.stderr.decode('utf-8')}"
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-5:]) > 0.80
