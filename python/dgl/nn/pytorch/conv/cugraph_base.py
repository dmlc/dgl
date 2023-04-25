"""An abstract base class for cugraph-ops nn module."""
import torch
from torch import nn


class CuGraphBaseConv(nn.Module):
    r"""An abstract base class for cugraph-ops nn module."""

    def __init__(self):
        super().__init__()
        self._cached_offsets_fg = None

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        raise NotImplementedError

    def forward(self, *args):
        r"""Runs the forward pass of the module."""
        raise NotImplementedError

    def pad_offsets(self, offsets: torch.Tensor, size: int) -> torch.Tensor:
        r"""Pad zero-in-degree nodes to the end of offsets to reach size. This
        is used to augment offset tensors from DGL blocks (MFGs) to be
        compatible with cugraph-ops full-graph primitives.

        Parameters
        ----------
        offsets :
            The (monotonically increasing) index pointer array in a CSC-format
            graph.
        size : int
            The length of offsets after padding.
        Returns
        -------
        torch.Tensor
            The augmented offsets array.
        """
        if self._cached_offsets_fg is None:
            self._cached_offsets_fg = torch.empty(
                size, dtype=offsets.dtype, device=offsets.device
            )
        elif self._cached_offsets_fg.numel() < size:
            self._cached_offsets_fg.resize_(size)

        self._cached_offsets_fg[: offsets.numel()] = offsets
        self._cached_offsets_fg[offsets.numel() : size] = offsets[-1]

        return self._cached_offsets_fg[:size]
