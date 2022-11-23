import tempfile
import os
import pytest
import numpy as np
from distpartitioning import array_readwriter

@pytest.mark.parametrize("shape", [[500], [300,10], [200,5,5], [100,5,5,5]])
@pytest.mark.parametrize("format", ['numpy', 'parquet'])
def test_array_readwriter(format, shape):
    original_arr = np.random.rand(*shape)
    fmt_meta = {"name": format}
    
    with tempfile.TemporaryDirectory() as test_dir:
        path = os.path.join(test_dir, f'nodes.{format}')
        array_readwriter.get_array_parser(
            **fmt_meta
        ).write(path, original_arr)
        arr = array_readwriter.get_array_parser(
            **fmt_meta
        ).read(path)

        assert original_arr.shape == arr.shape
        assert np.array_equal(original_arr, arr)
