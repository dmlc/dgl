from pathlib import Path
from jinja2 import Template
import copy
import typer
from pydantic import BaseModel, Field
from typing import Optional
import yaml
from ...utils.factory import PipelineFactory, NodeModelFactory, PipelineBase, DataFactory
from ...utils.base_model import EarlyStopConfig, DeviceEnum
from ...utils.yaml_dump import deep_convert_dict
import ruamel.yaml
from ruamel.yaml.comments import CommentedMap

class NodepredPipelineCfg(BaseModel):
    node_embed_size: Optional[int] = -1
    early_stop: Optional[EarlyStopConfig] = EarlyStopConfig()
    num_epochs: int = 200
    eval_period: int = 5
    optimizer: dict = {"name": "Adam", "lr": 0.005}
    loss: str = "CrossEntropyLoss"
@PipelineFactory.register("nodepred")
class NodepredPipeline(PipelineBase):

    user_cfg_cls = None

    def __init__(self):
        self.pipeline_name = "nodepred"

    @classmethod
    def setup_user_cfg_cls(cls):
        from ...utils.enter_config import UserConfig
        class NodePredUserConfig(UserConfig):
            data: DataFactory.get_pydantic_config() = Field(..., discriminator="name")
            model : NodeModelFactory.get_pydantic_model_config() = Field(..., discriminator="name")   
            general_pipeline: NodepredPipelineCfg = NodepredPipelineCfg()

        cls.user_cfg_cls = NodePredUserConfig
    
    @property
    def user_cfg_cls(self):
        return self.__class__.user_cfg_cls

    def get_cfg_func(self):
        def config(
            data: DataFactory.get_dataset_enum() = typer.Option(..., help="input data name"),
            cfg: str = typer.Option(
                "cfg.yml", help="output configuration path"),
            model: NodeModelFactory.get_model_enum() = typer.Option(..., help="Model name"),
            device: DeviceEnum = typer.Option("cpu", help="Device, cpu or cuda"),
        ):  
            self.__class__.setup_user_cfg_cls()
            generated_cfg = {
                "pipeline_name": "nodepred",
                "device": device,
                "data": {"name": data.name},
                "model": {"name": model.value},
                "general_pipeline": NodepredPipelineCfg()
            }
            output_cfg = self.user_cfg_cls(**generated_cfg).dict()
            comment_dict = deep_convert_dict(output_cfg)
            doc_dict = NodeModelFactory.get_constructor_doc_dict(model.value)
            for k, v in doc_dict.items():
                comment_dict["model"].yaml_add_eol_comment(v, key=k, column=30)

            yaml = ruamel.yaml.YAML()
            yaml.dump(comment_dict, Path(cfg).open("w"))

        return config

    @classmethod
    def gen_script(cls, user_cfg_dict):
        # Check validation
        cls.setup_user_cfg_cls()
        user_cfg = cls.user_cfg_cls(**user_cfg_dict)
        file_current_dir = Path(__file__).resolve().parent
        with open(file_current_dir / "nodepred.jinja-py", "r") as f:
            template = Template(f.read())

        render_cfg = copy.deepcopy(user_cfg_dict)
        model_code = NodeModelFactory.get_source_code(
            user_cfg_dict["model"]["name"])
        render_cfg["model_code"] = model_code
        render_cfg["model_class_name"] = NodeModelFactory.get_model_class_name(
            user_cfg_dict["model"]["name"])        
        render_cfg.update(DataFactory.get_generated_code_dict(user_cfg_dict["data"]["name"], '**cfg["data"]'))

        generated_user_cfg = copy.deepcopy(user_cfg_dict)
        if len(generated_user_cfg["data"]) == 1:
            generated_user_cfg.pop("data")
        else:
            generated_user_cfg["data"].pop("name")
        generated_user_cfg.pop("pipeline_name")
        generated_user_cfg["model"].pop("name")
        generated_user_cfg["general_pipeline"]["optimizer"].pop("name")

        generated_train_cfg = copy.deepcopy(user_cfg_dict["general_pipeline"])
        generated_train_cfg["optimizer"].pop("name")

        render_cfg["user_cfg_str"] = f"cfg = {str(generated_user_cfg)}"
        render_cfg["user_cfg"] = user_cfg_dict
        return template.render(**render_cfg)

    @staticmethod
    def get_description() -> str:
        return "Node classification pipeline"