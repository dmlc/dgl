"""Ondisk metadata of GraphBolt."""

from enum import Enum
from typing import Any, Dict, List, Optional

import pydantic

from ..internal_utils import version


__all__ = [
    "OnDiskFeatureDataFormat",
    "OnDiskTVTSetData",
    "OnDiskTVTSet",
    "OnDiskFeatureDataDomain",
    "OnDiskFeatureData",
    "OnDiskMetaData",
    "OnDiskGraphTopologyType",
    "OnDiskGraphTopology",
    "OnDiskTaskData",
]


class ExtraMetaData(pydantic.BaseModel, extra="allow"):
    """Group extra fields into metadata. Internal use only."""

    extra_fields: Optional[Dict[str, Any]] = {}

    # As pydantic 2.0 has changed the API of validators, we need to use
    # different validators for different versions to be compatible with
    # previous versions.
    if version.parse(pydantic.__version__) >= version.parse("2.0"):

        @pydantic.model_validator(mode="before")
        @classmethod
        def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
            """Build extra fields."""
            for key in list(values.keys()):
                if key not in cls.model_fields:
                    values["extra_fields"] = values.get("extra_fields", {})
                    values["extra_fields"][key] = values.pop(key)
            return values

    else:

        @pydantic.root_validator(pre=True)
        @classmethod
        def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
            """Build extra fields."""
            for key in list(values.keys()):
                if key not in cls.__fields__:
                    values["extra_fields"] = values.get("extra_fields", {})
                    values["extra_fields"][key] = values.pop(key)
            return values


class OnDiskFeatureDataFormat(str, Enum):
    """Enum of data format."""

    TORCH = "torch"
    NUMPY = "numpy"


class OnDiskTVTSetData(pydantic.BaseModel):
    """Train-Validation-Test set data."""

    name: Optional[str] = None
    format: OnDiskFeatureDataFormat
    in_memory: Optional[bool] = True
    path: str


class OnDiskTVTSet(pydantic.BaseModel):
    """Train-Validation-Test set."""

    type: Optional[str] = None
    data: List[OnDiskTVTSetData]


class OnDiskFeatureDataDomain(str, Enum):
    """Enum of feature data domain."""

    NODE = "node"
    EDGE = "edge"
    GRAPH = "graph"


class OnDiskFeatureData(ExtraMetaData):
    r"""The description of an on-disk feature."""
    domain: OnDiskFeatureDataDomain
    type: Optional[str] = None
    name: str
    format: OnDiskFeatureDataFormat
    path: str
    in_memory: Optional[bool] = True


class OnDiskGraphTopologyType(str, Enum):
    """Enum of graph topology type."""

    FUSED_CSC_SAMPLING = "FusedCSCSamplingGraph"


class OnDiskGraphTopology(pydantic.BaseModel):
    """The description of an on-disk graph topology."""

    type: OnDiskGraphTopologyType
    path: str


class OnDiskTaskData(ExtraMetaData):
    """Task specification in YAML."""

    train_set: Optional[List[OnDiskTVTSet]] = []
    validation_set: Optional[List[OnDiskTVTSet]] = []
    test_set: Optional[List[OnDiskTVTSet]] = []


class OnDiskMetaData(pydantic.BaseModel):
    """Metadata specification in YAML.

    As multiple node/edge types and multiple splits are supported, each TVT set
    is a list of list of ``OnDiskTVTSet``.
    """

    dataset_name: Optional[str] = None
    graph_topology: Optional[OnDiskGraphTopology] = None
    feature_data: Optional[List[OnDiskFeatureData]] = []
    tasks: Optional[List[OnDiskTaskData]] = []
