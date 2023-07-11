"""Ondisk metadata of GraphBolt."""

from typing import List, Optional

import pydantic
import pydantic_yaml


__all__ = [
    "OnDiskFeatureDataFormat",
    "OnDiskTVTSet",
    "OnDiskFeatureDataDomain",
    "OnDiskFeatureData",
    "OnDiskMetaData",
    "OnDiskGraphTopologyType",
    "OnDiskGraphTopology",
]


class OnDiskFeatureDataFormat(pydantic_yaml.YamlStrEnum):
    """Enum of data format."""

    TORCH = "torch"
    NUMPY = "numpy"


class OnDiskTVTSet(pydantic.BaseModel):
    """Train-Validation-Test set."""

    type: Optional[str]
    format: OnDiskFeatureDataFormat
    in_memory: Optional[bool] = True
    path: str


class OnDiskFeatureDataDomain(pydantic_yaml.YamlStrEnum):
    """Enum of feature data domain."""

    NODE = "node"
    EDGE = "edge"
    GRAPH = "graph"


class OnDiskFeatureData(pydantic.BaseModel):
    r"""The description of an on-disk feature."""
    domain: OnDiskFeatureDataDomain
    type: Optional[str]
    name: str
    format: OnDiskFeatureDataFormat
    path: str
    in_memory: Optional[bool] = True


class OnDiskGraphTopologyType(pydantic_yaml.YamlStrEnum):
    """Enum of graph topology type."""

    CSC_SAMPLING = "CSCSamplingGraph"


class OnDiskGraphTopology(pydantic.BaseModel):
    """The description of an on-disk graph topology."""

    type: OnDiskGraphTopologyType
    path: str


class OnDiskMetaData(pydantic_yaml.YamlModel):
    """Metadata specification in YAML.

    As multiple node/edge types and multiple splits are supported, each TVT set
    is a list of list of ``OnDiskTVTSet``.
    """

    dataset_name: Optional[str] = None
    num_classes: Optional[int] = None
    num_labels: Optional[int] = None
    graph_topology: Optional[OnDiskGraphTopology] = None
    feature_data: Optional[List[OnDiskFeatureData]] = []
    train_sets: Optional[List[List[OnDiskTVTSet]]] = []
    validation_sets: Optional[List[List[OnDiskTVTSet]]] = []
    test_sets: Optional[List[List[OnDiskTVTSet]]] = []
