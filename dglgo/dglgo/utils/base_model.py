import copy
import enum
from enum import Enum, IntEnum
from typing import Optional

from jinja2 import Template
from pydantic import (
    BaseModel as PydanticBaseModel,
    create_model,
    create_model,
    Field,
)


class DeviceEnum(str, Enum):
    cpu = "cpu"
    cuda = "cuda"


class DGLBaseModel(PydanticBaseModel):
    class Config:
        extra = "allow"
        use_enum_values = True

    @classmethod
    def with_fields(cls, model_name, **field_definitions):
        return create_model(model_name, __base__=cls, **field_definitions)


def get_literal_value(type_):
    if hasattr(type_, "__values__"):
        name = type_.__values__[0]
    elif hasattr(type_, "__args__"):
        name = type_.__args__[0]
    return name


def extract_name(union_type):
    name_dict = {}
    for t in union_type.__args__:
        type_ = t.__fields__["name"].type_
        name = get_literal_value(type_)
        name_dict[name] = name
    return enum.Enum("Choice", name_dict)


class EarlyStopConfig(DGLBaseModel):
    patience: int = 20
    checkpoint_path: str = "checkpoint.pth"
