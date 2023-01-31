import argparse
import importlib.util
import logging
import os


def prepare_env():
    # restart ssh service to enable port 2233
    os.system("service ssh restart")

    # install latest DGL nightly build
    cuda = os.environ.get("CUDA", None)
    if cuda is None:
        install_cmd = (
            "pip3 install --pre dgl "
            "-f https://data.dgl.ai/wheels-test/repo.html"
        )
    else:
        install_cmd = (
            "pip3 install --pre dgl "
            f"-f https://data.dgl.ai/wheels-test/{cuda}/repo.html"
        )
    os.system(install_cmd)
    os.system("python3 -c 'import dgl;print(dgl.__version__)'")
    logging.info(f"Latest DGL nightly build[CUDA: {cuda}] is installed...")

    # check and define required envs
    os.environ["DGL_ROOT_DIR"] = "/dgl"
    workspace = os.environ.get("WORKSPACE", "/workspace")
    if not os.path.isdir(workspace):
        os.mkdir(workspace)
    os.environ["WORKSPACE"] = workspace
    os.environ["IP_CONFIG"] = os.path.join(workspace, "ip_config.txt")
    os.environ["SSH_PORT"] = "2233"
    if (
        os.environ["AWS_BATCH_JOB_MAIN_NODE_INDEX"]
        == os.environ["AWS_BATCH_JOB_NODE_INDEX"]
    ):
        os.environ["NODE_TYPE"] = "MAIN_NODE"
    else:
        os.environ["NODE_TYPE"] = "CHILD_NODE"

    # generate ip_config.txt
    bin_path = os.path.join(
        os.environ["DGL_ROOT_DIR"], "tests/regression/dist_env_setup.sh"
    )
    os.system(f"bash {bin_path}")


def create_task(task_type):
    task_mod = importlib.import_module("task")
    return getattr(task_mod, task_type)()


if __name__ == "__main__":
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO)
    logging.info("-------------------------- DistTestLauncher -------------")

    parser = argparse.ArgumentParser(
        description="Distributed test launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task", type=str, required=True, help="task type: partition or train"
    )
    args, _ = parser.parse_known_args()

    # prepare distributed compute environment
    prepare_env()

    # run partition or train test
    task = create_task(args.task)
    task.run()

    logging.info("Dist test launcher is done...")
