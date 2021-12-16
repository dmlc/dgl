from typing import Optional
import yaml
import jinja2
from jinja2 import Template
from enum import Enum, IntEnum
import copy
from pydantic import BaseModel, ValidationError

class DatasetEnum(str, Enum):
    RedditDataset = "RedditDataset"
    CoraGraphDataset = "CoraGraphDataset"

class PipelineEnum(str, Enum):
    nodepred = "nodepred"
    nodepred_ns = 'nodepred_ns'

class DataConfig(BaseModel):
    name: DatasetEnum
    class Config:
        extra = "allow"


ModelConfig = dict
# TrainConfig = dict
def enum_representer(dumper: yaml.Dumper, data):
    return dumper.represent_scalar(u'tag:yaml.org,2002:str', data.value)

yaml.SafeDumper.add_representer(PipelineEnum, enum_representer)
yaml.SafeDumper.add_representer(DatasetEnum, enum_representer)
print("register")
# yaml.add_multi_representer
class PipelineConfig(BaseModel):    
    node_embed_size: Optional[int] = -1
    early_stop: Optional[dict]
    num_epochs: int = 200
    eval_period: int = 5
    optimizer: dict = {"name": "Adam", "lr": 0.005}
    loss: str = "CrossEntropyLoss"

# class TrainConfig(BaseModel):
#     sampler: Optional[dict]
#     # optimizer: dict
#     early_stop: dict
#     loss: str

class UserConfig(BaseModel):
    version: Optional[str] = "0.0.1"
    pipeline_name: PipelineEnum
    device: str = "cpu"
    data: DataConfig
    model: ModelConfig
    general_pipeline: PipelineConfig = PipelineConfig()

