import os
import subprocess
import sys

EXAMPLE_ROOT = os.path.join(
    os.path.dirname(os.path.relpath(__file__)),
    "..",
    "..",
    "examples",
    "sampling",
    "graphbolt",
    "quickstart",
)


def test_gcn():
    script = os.path.join(EXAMPLE_ROOT, "gcn.py")
    out = subprocess.run(["python", str(script)], capture_output=True)
    assert out.returncode == 0
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-5:]) > 0.70
