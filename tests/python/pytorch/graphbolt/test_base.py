import re
import unittest

import backend as F

import dgl.graphbolt as gb
import pytest
import torch


@unittest.skipIf(F._default_context_str == "cpu", "CopyTo needs GPU to test")
def test_CopyTo():
    dp = gb.ItemSampler(torch.randn(20), 4)
    dp = gb.CopyTo(dp, "cuda")

    for data in dp:
        assert data.device.type == "cuda"


def test_etype_tuple_to_str():
    """Convert etype from tuple to string."""
    # Test for expected input.
    c_etype = ("user", "like", "item")
    c_etype_str = gb.etype_tuple_to_str(c_etype)
    assert c_etype_str == "user:like:item"

    # Test for unexpected input: not a tuple.
    c_etype = "user:like:item"
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Passed-in canonical etype should be in format of (str, str, str). "
            "But got user:like:item."
        ),
    ):
        _ = gb.etype_tuple_to_str(c_etype)

    # Test for unexpected input: tuple with wrong length.
    c_etype = ("user", "like")
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Passed-in canonical etype should be in format of (str, str, str). "
            "But got ('user', 'like')."
        ),
    ):
        _ = gb.etype_tuple_to_str(c_etype)


def test_etype_str_to_tuple():
    """Convert etype from string to tuple."""
    # Test for expected input.
    c_etype_str = "user:like:item"
    c_etype = gb.etype_str_to_tuple(c_etype_str)
    assert c_etype == ("user", "like", "item")

    # Test for unexpected input: string with wrong format.
    c_etype_str = "user:like"
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Passed-in canonical etype should be in format of 'str:str:str'. "
            "But got user:like."
        ),
    ):
        _ = gb.etype_str_to_tuple(c_etype_str)
