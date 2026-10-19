import pytest
import torch

from concat_impl import concat_impl


@pytest.mark.parametrize(
    "shapes,dim",
    [
        ([(256, 1024), (256, 1024)], 0),
        ([(1024, 512), (1024, 512)], -1),
        ([(128, 64)] * 4, 0),
        ([(7,), (11,), (3,)], 0),
        ([(4, 1, 32), (4, 2, 32), (4, 5, 32)], 1),
        ([(2, 3, 4, 5), (2, 3, 7, 5)], 2),
        ([(3, 5)], 1),
    ],
)
def test_concat_shapes(shapes, dim):
    tensors = [torch.randn(shape) for shape in shapes]
    actual = concat_impl(tensors, dim)
    expected = torch.cat(tensors, dim=dim)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.device.type == "cpu"


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.float32, torch.float64, torch.int32, torch.int64],
)
def test_concat_dtypes(dtype):
    if dtype.is_floating_point:
        tensors = [torch.randn(3, 4, dtype=dtype), torch.randn(3, 2, dtype=dtype)]
    else:
        tensors = [
            torch.randint(-20, 20, (3, 4), dtype=dtype),
            torch.randint(-20, 20, (3, 2), dtype=dtype),
        ]

    actual = concat_impl(tensors, 1)
    expected = torch.cat(tensors, dim=1)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == dtype


@pytest.mark.parametrize(
    "tensors,dim",
    [
        ([torch.empty(2, 0, 4), torch.randn(2, 3, 4)], 1),
        ([torch.empty(0, 2, 4), torch.empty(0, 3, 4)], 1),
    ],
)
def test_concat_empty_dimensions(tensors, dim):
    actual = concat_impl(tensors, dim)
    expected = torch.cat(tensors, dim=dim)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "tensors,dim",
    [
        ([], 0),
        ([torch.randn(2, 3), torch.randn(2, 3)], 2),
        ([torch.randn(2, 3), torch.randn(4, 2)], 0),
        ([torch.randn(2, 3), torch.ones(2, 3, dtype=torch.int32)], 0),
    ],
)
def test_concat_rejects_invalid_inputs(tensors, dim):
    with pytest.raises(Exception):
        concat_impl(tensors, dim)
