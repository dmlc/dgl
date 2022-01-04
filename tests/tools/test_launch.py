import unittest

from tools.launch import wrap_udf_in_torch_dist_launcher, wrap_cmd_with_local_envvars, construct_dgl_server_env_vars, \
    construct_dgl_client_env_vars


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
        expected = "python3.7 -m torch.distributed.launch " \
                   "--nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr=127.0.0.1 " \
                   "--master_port=1234 path/to/some/trainer.py arg1 arg2"
        self.assertEqual(wrapped_udf_command, expected)

    def test_chained_udf(self):
        # test that a chained udf_command is properly handled
        udf_command = "cd path/to && python3.7 path/to/some/trainer.py arg1 arg2"
        wrapped_udf_command = wrap_udf_in_torch_dist_launcher(
            udf_command=udf_command,
            num_trainers=2,
            num_nodes=2,
            node_rank=1,
            master_addr="127.0.0.1",
            master_port=1234,
        )
        expected = "cd path/to && python3.7 -m torch.distributed.launch " \
                   "--nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr=127.0.0.1 " \
                   "--master_port=1234 path/to/some/trainer.py arg1 arg2"
        self.assertEqual(wrapped_udf_command, expected)

    def test_py_versions(self):
        # test that this correctly handles different py versions/binaries
        py_binaries = (
            "python3.7", "python3.8", "python3.9", "python3", "python"
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
            expected = "{python_bin} -m torch.distributed.launch ".format(python_bin=py_bin) + \
                       "--nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr=127.0.0.1 " \
                       "--master_port=1234 path/to/some/trainer.py arg1 arg2"
            self.assertEqual(wrapped_udf_command, expected)


class TestWrapCmdWithLocalEnvvars(unittest.TestCase):
    """wrap_cmd_with_local_envvars()"""

    def test_simple(self):
        self.assertEqual(
            wrap_cmd_with_local_envvars("ls && pwd", "VAR1=value1 VAR2=value2"),
            "(export VAR1=value1 VAR2=value2; ls && pwd)"
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
                graph_format="csc"
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
            )
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
                pythonpath="some/pythonpath/"
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
                "PYTHONPATH=some/pythonpath/ "
            )
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
            )
        )


if __name__ == '__main__':
    unittest.main()
