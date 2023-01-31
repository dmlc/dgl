import logging
import os
import pathlib
import time


class Task:
    def __init__(self):
        pass

    def run(self):
        self._prepare_data()
        self._continue_on_main_node_only()
        self._do_run()
        self._print_metrics()

    def _prepare_data(self):
        raise RuntimeError("Not implemented...")

    def _continue_on_main_node_only(self):
        workspace = os.environ["WORKSPACE"]
        sync_file = os.path.join(workspace, "dist_dgl_node_ready.sync")
        pathlib.Path(sync_file).touch()
        if (
            os.environ["AWS_BATCH_JOB_MAIN_NODE_INDEX"]
            != os.environ["AWS_BATCH_JOB_NODE_INDEX"]
        ):
            logging.info("Child node goes to sleep now...")
            time.sleep(60 * 60 * 24)
        ip_config = os.environ["IP_CONFIG"]
        ssh_port = os.environ["SSH_PORT"]
        with open(ip_config, "r") as f:
            for line in f:
                ip = line.rstrip()
                logging.info(f"-------IP: {ip}")
                cmd = f"while [ ! -f {sync_file} ]; do sleep 2; done"
                os.system(
                    f"ssh -o StrictHostKeyChecking=no -p {ssh_port} "
                    f"{ip} '{cmd}'"
                )
                logging.info(f"{ip} is in sleep now...")
        logging.info("Main node continues...")

    def _do_run(self):
        raise RuntimeError("Not implemented...")

    def _print_metrics(self):
        raise RuntimeError("Not implemented...")
