import json
from typing import Optional

import pydantic as dt
from dgl import DGLError


class PartitionMeta(dt.BaseModel):
    """Metadata that describes the partition assignment results.

    Regardless of the choice of partitioning algorithm, a metadata JSON file
    will be created in the output directory which includes the meta information
    of the partition algorithm.

    To generate a metadata JSON:

    >>> part_meta = PartitionMeta(version='1.0.0', num_parts=4, algo_name='random')
    >>> with open('metadata.json', 'w') as f:
    ...     json.dump(part_meta.dict(), f)

    To read a metadata JSON:

    >>> with open('metadata.json') as f:
    ...     part_meta = PartitionMeta(**(json.load(f)))

    """

    # version of metadata JSON.
    version: Optional[str] = "1.0.0"
    # number of partitions.
    num_parts: int
    # name of partition algorithm.
    algo_name: str


def dump_partition_meta(part_meta, meta_file):
    """Dump partition metadata into json file.

    Parameters
    ----------
    part_meta : PartitionMeta
        The partition metadata.
    meta_file : str
        The target file to save data.
    """
    with open(meta_file, "w") as f:
        json.dump(part_meta.dict(), f, sort_keys=True, indent=4)


def load_partition_meta(meta_file):
    """Load partition metadata and do sanity check.

    Parameters
    ----------
    meta_file : str
        The path of the partition metadata file.

    Returns
    -------
    PartitionMeta
        The partition metadata.
    """
    with open(meta_file) as f:
        try:
            part_meta = PartitionMeta(**(json.load(f)))
        except dt.ValidationError as e:
            raise DGLError(
                f"Invalid partition metadata JSON. Error details: {e.json()}"
            )
        if part_meta.version != "1.0.0":
            raise DGLError(
                f"Invalid version[{part_meta.version}]. Supported versions: '1.0.0'"
            )
        if part_meta.num_parts <= 0:
            raise DGLError(
                f"num_parts[{part_meta.num_parts}] should be greater than 0."
            )
        if part_meta.algo_name not in ["random", "metis"]:
            raise DGLError(
                f"algo_name[{part_meta.num_parts}] is not supported."
            )
        return part_meta
