import unittest

from tools.launch import wrap_udf_in_torch_dist_launcher


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


if __name__ == '__main__':
    unittest.main()
