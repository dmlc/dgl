from typing import Optional
import yaml
import jinja2
from jinja2 import Template
from enum import Enum, IntEnum
import copy
from pydantic import BaseModel, ValidationError


class PipelineEnum(str, Enum):
    nodepred = "nodepred"
    nodepred_ns = 'nodepred_ns'

DataConfig = dict
ModelConfig = dict
# TrainConfig = dict



class TrainConfig(BaseModel):
    sampler: Optional[dict]
    optimizer: dict
    early_stop: dict
    loss: str

class UserConfig(BaseModel):
    version: Optional[str] = "0.0.1"
    pipeline_name: PipelineEnum
    data: DataConfig
    node_embed_size: Optional[int] = -1
    model: ModelConfig
    train_config: TrainConfig

