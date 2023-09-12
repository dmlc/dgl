import json
import os
import tempfile
import unittest

from launch import *


class TestWrapUdfInTorchDistLauncher(unittest.TestCase):
    """wrap_udf_in_torch_dist_launcher()"""

    def test_simple(self):
        # test that a simple udf_command is correctly wrapped
        udf_command = "python3.7 path/to/some/trainer.py arg1 arg2"
        wrapped_udf_command = wrap_udf_in_torch_dist_launcher(
            udf_command=udf_command,
            num_trainers=2,
            num_nodes=2,
            node_rank=1,
            master_addr="127.0.0.1",
            master_port=1234,
        )
        expected = (
            "python3.7 -m torch.distributed.run "
            "--nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr=127.0.0.1 "
            "--master_port=1234 path/to/some/trainer.py arg1 arg2"
        )
        self.assertEqual(wrapped_udf_command, expected)

    def test_chained_udf(self):
        # test that a chained udf_command is properly handled
        udf_command = (
            "cd path/to && python3.7 path/to/some/trainer.py arg1 arg2"
        )
        wrapped_udf_command = wrap_udf_in_torch_dist_launcher(
            udf_command=udf_command,
            num_trainers=2,
            num_nodes=2,
            node_rank=1,
            master_addr="127.0.0.1",
            master_port=1234,
        )
        expected = (
            "cd path/to && python3.7 -m torch.distributed.run "
            "--nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr=127.0.0.1 "
            "--master_port=1234 path/to/some/trainer.py arg1 arg2"
        )
        self.assertEqual(wrapped_udf_command, expected)

    def test_py_versions(self):
        # test that this correctly handles different py versions/binaries
        py_binaries = (
            "python3.7",
            "python3.8",
            "python3.9",
            "python3",
            "python",
        )
        udf_command = "{python_bin} path/to/some/trainer.py arg1 arg2"

        for py_bin in py_binaries:
            wrapped_udf_command = wrap_udf_in_torch_dist_launcher(
                udf_command=udf_command.format(python_bin=py_bin),
                num_trainers=2,
                num_nodes=2,
                node_rank=1,
                master_addr="127.0.0.1",
                master_port=1234,
            )
            expected = (
                "{python_bin} -m torch.distributed.run ".format(
                    python_bin=py_bin
                )
                + "--nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr=127.0.0.1 "
                "--master_port=1234 path/to/some/trainer.py arg1 arg2"
            )
            self.assertEqual(wrapped_udf_command, expected)


class TestWrapCmdWithLocalEnvvars(unittest.TestCase):
    """wrap_cmd_with_local_envvars()"""

    def test_simple(self):
        self.assertEqual(
            wrap_cmd_with_local_envvars("ls && pwd", "VAR1=value1 VAR2=value2"),
            "(export VAR1=value1 VAR2=value2; ls && pwd)",
        )


class TestConstructDglServerEnvVars(unittest.TestCase):
    """construct_dgl_server_env_vars()"""

    def test_simple(self):
        self.assertEqual(
            construct_dgl_server_env_vars(
                num_samplers=2,
                num_server_threads=3,
                tot_num_clients=4,
                part_config="path/to/part.config",
                ip_config="path/to/ip.config",
                num_servers=5,
                graph_format="csc",
            ),
            (
                "DGL_ROLE=server "
                "DGL_NUM_SAMPLER=2 "
                "OMP_NUM_THREADS=3 "
                "DGL_NUM_CLIENT=4 "
                "DGL_CONF_PATH=path/to/part.config "
                "DGL_IP_CONFIG=path/to/ip.config "
                "DGL_NUM_SERVER=5 "
                "DGL_GRAPH_FORMAT=csc "
            ),
        )


class TestConstructDglClientEnvVars(unittest.TestCase):
    """construct_dgl_client_env_vars()"""

    def test_simple(self):
        # with pythonpath
        self.assertEqual(
            construct_dgl_client_env_vars(
                num_samplers=1,
                tot_num_clients=2,
                part_config="path/to/part.config",
                ip_config="path/to/ip.config",
                num_servers=3,
                graph_format="csc",
                num_omp_threads=4,
                group_id=0,
                pythonpath="some/pythonpath/",
            ),
            (
                "DGL_DIST_MODE=distributed "
                "DGL_ROLE=client "
                "DGL_NUM_SAMPLER=1 "
                "DGL_NUM_CLIENT=2 "
                "DGL_CONF_PATH=path/to/part.config "
                "DGL_IP_CONFIG=path/to/ip.config "
                "DGL_NUM_SERVER=3 "
                "DGL_GRAPH_FORMAT=csc "
                "OMP_NUM_THREADS=4 "
                "DGL_GROUP_ID=0 "
                "PYTHONPATH=some/pythonpath/ "
            ),
        )
        # without pythonpath
        self.assertEqual(
            construct_dgl_client_env_vars(
                num_samplers=1,
                tot_num_clients=2,
                part_config="path/to/part.config",
                ip_config="path/to/ip.config",
                num_servers=3,
                graph_format="csc",
                num_omp_threads=4,
                group_id=0,
            ),
            (
                "DGL_DIST_MODE=distributed "
                "DGL_ROLE=client "
                "DGL_NUM_SAMPLER=1 "
                "DGL_NUM_CLIENT=2 "
                "DGL_CONF_PATH=path/to/part.config "
                "DGL_IP_CONFIG=path/to/ip.config "
                "DGL_NUM_SERVER=3 "
                "DGL_GRAPH_FORMAT=csc "
                "OMP_NUM_THREADS=4 "
                "DGL_GROUP_ID=0 "
            ),
        )


def test_submit_jobs():
    class Args:
        pass

    args = Args()

    with tempfile.TemporaryDirectory() as test_dir:
        num_machines = 8
        ip_config = os.path.join(test_dir, "ip_config.txt")
        with open(ip_config, "w") as f:
            for i in range(num_machines):
                f.write("{} {}\n".format("127.0.0." + str(i), 30050))
        part_config = os.path.join(test_dir, "ogb-products.json")
        with open(part_config, "w") as f:
            json.dump({"num_parts": num_machines}, f)
        args.num_trainers = 8
        args.num_samplers = 1
        args.num_servers = 4
        args.workspace = test_dir
        args.part_config = "ogb-products.json"
        args.ip_config = "ip_config.txt"
        args.num_server_threads = 1
        args.graph_format = "csc"
        args.extra_envs = ["NCCL_DEBUG=INFO"]
        args.num_omp_threads = 1
        udf_command = "python3 train_dist.py --num_epochs 10"
        clients_cmd, servers_cmd = submit_jobs(args, udf_command, dry_run=True)

        def common_checks():
            assert "cd " + test_dir in cmd
            assert "export " + args.extra_envs[0] in cmd
            assert f"DGL_NUM_SAMPLER={args.num_samplers}" in cmd
            assert (
                f"DGL_NUM_CLIENT={args.num_trainers*(args.num_samplers+1)*num_machines}"
                in cmd
            )
            assert f"DGL_CONF_PATH={args.part_config}" in cmd
            assert f"DGL_IP_CONFIG={args.ip_config}" in cmd
            assert f"DGL_NUM_SERVER={args.num_servers}" in cmd
            assert f"DGL_GRAPH_FORMAT={args.graph_format}" in cmd
            assert f"OMP_NUM_THREADS={args.num_omp_threads}" in cmd
            assert udf_command[len("python3 ") :] in cmd

        for cmd in clients_cmd:
            common_checks()
            assert "DGL_DIST_MODE=distributed" in cmd
            assert "DGL_ROLE=client" in cmd
            assert "DGL_GROUP_ID=0" in cmd
            assert (
                f"python3 -m torch.distributed.run --nproc_per_node={args.num_trainers} --nnodes={num_machines}"
                in cmd
            )
            assert "--master_addr=127.0.0" in cmd
            assert "--master_port=1234" in cmd
        for cmd in servers_cmd:
            common_checks()
            assert "DGL_ROLE=server" in cmd
            assert "DGL_SERVER_ID=" in cmd


if __name__ == "__main__":
    unittest.main()
