from typing import Optional
import yaml
import jinja2
from jinja2 import Template
from enum import Enum, IntEnum
import copy
from pydantic import BaseModel as PydanticBaseModel
from .pipeline import nodepred, nodepred_sample
from .utils.factory import PipelineFactory
class DatasetEnum(str, Enum):
    RedditDataset = "RedditDataset"
    CoraGraphDataset = "CoraGraphDataset"


class BaseModel(PydanticBaseModel):
    class Config:
        extra = "allow"
        use_enum_values = True


# class PipelineEnum(str, Enum):
#     nodepred = "nodepred"
#     nodepred_ns = 'nodepred_ns'
# PipelineEnum = PipelineFactory.get_pipeline_enum()

class DataConfig(BaseModel):
    name: DatasetEnum
    class Config:
        extra = "allow"

output_file_path = None

ModelConfig = dict

SamplerConfig = dict

class PipelineConfig(BaseModel):    
    node_embed_size: Optional[int] = -1
    early_stop: Optional[dict]
    num_epochs: int = 200
    eval_period: int = 5
    optimizer: dict = {"name": "Adam", "lr": 0.005}
    loss: str = "CrossEntropyLoss"

class UserConfig(BaseModel):
    version: Optional[str] = "0.0.1"
    pipeline_name: PipelineFactory.get_pipeline_enum()
    device: str = "cpu"
    data: DataConfig
    model: ModelConfig
    general_pipeline: PipelineConfig = PipelineConfig()

