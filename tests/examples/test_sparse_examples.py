import os
import subprocess
import sys

EXAMPLE_ROOT = os.path.join(
    os.path.dirname(os.path.relpath(__file__)),
    "..",
    "..",
    "examples",
    "sparse",
)


def test_gcn():
    script = os.path.join(EXAMPLE_ROOT, "gcn.py")
    out = subprocess.run(["python", str(script)], capture_output=True)
    assert out.returncode == 0
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-5:]) > 0.75


def test_gcnii():
    script = os.path.join(EXAMPLE_ROOT, "gcnii.py")
    out = subprocess.run(["python", str(script)], capture_output=True)
    assert out.returncode == 0
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-5:]) > 0.75


def test_appnp():
    script = os.path.join(EXAMPLE_ROOT, "appnp.py")
    out = subprocess.run(["python", str(script)], capture_output=True)
    assert out.returncode == 0
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-5:]) > 0.75


def test_c_and_s():
    script = os.path.join(EXAMPLE_ROOT, "c_and_s.py")
    out = subprocess.run(["python", str(script)], capture_output=True)
    assert out.returncode == 0
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-5:]) > 0.7


def test_gat():
    script = os.path.join(EXAMPLE_ROOT, "gat.py")
    out = subprocess.run(["python", str(script)], capture_output=True)
    assert out.returncode == 0
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-5:]) > 0.7


def test_hgnn():
    script = os.path.join(EXAMPLE_ROOT, "hgnn.py")
    out = subprocess.run(["python", str(script)], capture_output=True)
    assert out.returncode == 0
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-5:]) > 0.66


def test_hypergraphatt():
    script = os.path.join(EXAMPLE_ROOT, "hypergraphatt.py")
    out = subprocess.run(
        ["python", str(script), "--epochs=10"], capture_output=True
    )
    assert out.returncode == 0


def test_sgc():
    script = os.path.join(EXAMPLE_ROOT, "sgc.py")
    out = subprocess.run(["python", str(script)], capture_output=True)
    assert out.returncode == 0
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-5:]) > 0.7


def test_sign():
    script = os.path.join(EXAMPLE_ROOT, "sign.py")
    out = subprocess.run(["python", str(script)], capture_output=True)
    assert out.returncode == 0
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-5:]) > 0.7


def test_twirls():
    script = os.path.join(EXAMPLE_ROOT, "twirls.py")

    out = subprocess.run(["python", str(script)], capture_output=True)
    assert out.returncode == 0
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-5:]) > 0.7

    out = subprocess.run(
        ["python", str(script), "--attention"], capture_output=True
    )
    assert out.returncode == 0
    stdout = out.stdout.decode("utf-8")
    assert float(stdout[-5:]) > 0.65
