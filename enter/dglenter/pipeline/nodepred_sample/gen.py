from enum import Enum
from pathlib import Path
from typing import Optional, List, Literal, Union
from jinja2 import Template, ext
from pydantic import BaseModel, Field
import copy
import yaml

import typer
from ...utils.factory import PipelineFactory, NodeModelFactory, PipelineBase, DataFactory
from ...utils.base_model import extract_name, EarlyStopConfig, DeviceEnum

from ...utils.yaml_dump import deep_convert_dict
import ruamel.yaml
from ruamel.yaml.comments import CommentedMap


class MultiLayerSamplerConfig(BaseModel):
    name: Literal["neighbor"]
    fan_out: List[int] = [5, 10]
    batch_size: int = Field(64, description="Batch size")
    num_workers: int = 4
    eval_batch_size: int = 1024
    eval_num_workers: int = 4

    class Config:
        extra = 'forbid'


class OtherSamplerConfig(BaseModel):
    name: Literal["other"]
    demo_batch_size: int = Field(64, description="Batch size")

    class Config:
        extra = 'forbid'


SamplerConfig = Union[MultiLayerSamplerConfig,
                      OtherSamplerConfig]

SamplerChoice = extract_name(SamplerConfig)


class NodepredNSPipelineCfg(BaseModel):
    sampler: SamplerConfig = Field(..., discriminator='name')
    node_embed_size: Optional[int] = -1
    early_stop: Optional[EarlyStopConfig] = EarlyStopConfig()
    num_epochs: int = 200
    eval_period: int = 5
    optimizer: dict = {"name": "Adam", "lr": 0.005}
    loss: str = "CrossEntropyLoss"


@PipelineFactory.register("nodepred-ns")
class NodepredNsPipeline(PipelineBase):
    def __init__(self):
        self.pipeline_name = "nodepred-ns"
        self.default_cfg = None

    def get_cfg_func(self):
        def config(
            data: DataFactory.get_dataset_enum() = typer.Option(..., help="input data name"),
            cfg: str = typer.Option(
                "cfg.yml", help="output configuration path"),
            sampler: SamplerChoice = typer.Option(
                "neighbor", help="Specify sampler name"),
            model: NodeModelFactory.get_model_enum() = typer.Option(..., help="Model name"),
            device: DeviceEnum = typer.Option(
                "cpu", help="Device, cpu or cuda"),
        ):
            from ...utils.enter_config import UserConfig
            generated_cfg = {
                "pipeline_name": "nodepred-ns",
                "device": device,
                "data": {"name": data.name},
                "model": {"name": model.value},
                "general_pipeline": NodepredNSPipelineCfg(sampler={"name": sampler.value})
            }
            output_cfg = UserConfig(**generated_cfg).dict()
            output_cfg = self.user_cfg_cls(**generated_cfg).dict()
            comment_dict = deep_convert_dict(output_cfg)
            doc_dict = NodeModelFactory.get_constructor_doc_dict(model.value)
            for k, v in doc_dict.items():
                comment_dict["model"].yaml_add_eol_comment(v, key=k, column=30)

            yaml = ruamel.yaml.YAML()
            yaml.dump(comment_dict, Path(cfg).open("w"))
            print("Configuration file is generated at {}".format(
                Path(cfg).absolute()))

        return config

    # def get_default_cfg(self):
    #     return self.default_cfg

    @staticmethod
    def gen_script(user_cfg_dict):
        file_current_dir = Path(__file__).resolve().parent
        template_filename = file_current_dir / "nodepred-ns.jinja-py"
        with open(template_filename, "r") as f:
            template = Template(f.read())
        print(user_cfg_dict)
        pipeline_cfg = NodepredNSPipelineCfg(
            **user_cfg_dict["general_pipeline"])

        render_cfg = copy.deepcopy(user_cfg_dict)
        model_code = NodeModelFactory.get_source_code(
            user_cfg_dict["model"]["name"])
        render_cfg["model_code"] = model_code
        render_cfg["model_class_name"] = NodeModelFactory.get_model_class_name(
            user_cfg_dict["model"]["name"])
        render_cfg.update(DataFactory.get_generated_code_dict(
            user_cfg_dict["data"]["name"], '**cfg["data"]'))
        generated_user_cfg = copy.deepcopy(user_cfg_dict)

        if len(generated_user_cfg["data"]) == 1:
            generated_user_cfg.pop("data")
        else:
            generated_user_cfg["data"].pop("name")

        generated_user_cfg.pop("pipeline_name")
        generated_user_cfg["model"].pop("name")
        generated_user_cfg['general_pipeline']["optimizer"].pop("name")

        render_cfg["user_cfg_str"] = f"cfg = {str(generated_user_cfg)}"
        render_cfg["user_cfg"] = user_cfg_dict
        with open("output.py", "w") as f:
            return template.render(**render_cfg)

    @staticmethod
    def get_description() -> str:
        return "Node classification sampling pipeline"
