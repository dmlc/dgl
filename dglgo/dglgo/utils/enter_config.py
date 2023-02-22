import copy
from enum import Enum, IntEnum
from typing import Optional

import jinja2
import yaml
from jinja2 import Template
from pydantic import BaseModel as PydanticBaseModel, create_model, Field

from .base_model import DGLBaseModel

# from ..pipeline import nodepred, nodepred_sample
from .factory import DataFactory, ModelFactory, PipelineFactory


class PipelineConfig(DGLBaseModel):
    node_embed_size: Optional[int] = -1
    early_stop: Optional[dict]
    num_epochs: int = 200
    eval_period: int = 5
    optimizer: dict = {"name": "Adam", "lr": 0.005}
    loss: str = "CrossEntropyLoss"


class UserConfig(DGLBaseModel):
    version: Optional[str] = "0.0.2"
    pipeline_name: PipelineFactory.get_pipeline_enum()
    pipeline_mode: str
    device: str = "cpu"
