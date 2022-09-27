from typing import Optional
import pydantic as dt
import json
from dgl import DGLError

class PartitionMeta(dt.BaseModel):
    """ Describe fields of partition metadata.
    """
    version: Optional[str] = '1.0.0'
    num_parts: int

def dump_partition_meta(part_meta, meta_file):
    """ Dump partition metadata into json file.

    Parameters
    ----------
    part_meta : PartitionMeta
        The partition metadata.
    meta_file : str
        The target file to save data.
    """
    with open(meta_file, 'w') as f:
        json.dump(part_meta.dict(), f, sort_keys=True, indent=4)

def load_partition_meta(meta_file):
    """ Load partition metadata and do sanity check.

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
            print(
                "Details of pydantic.ValidationError:\n{}".format(e.json()))
            raise DGLError(
                "Validation Error for YAML fields. Details are shown above.")
        if part_meta.version != '1.0.0':
            raise DGLError(
                f"Invalid version[{part_meta.version}]. Supported versions: '1.0.0'")
        if part_meta.num_parts <= 0:
            raise DGLError(
                f"num_parts[{part_meta.num_parts}] should be greater than 0.")
        return part_meta
