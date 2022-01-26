from pathlib import Path
from jinja2 import Template
import copy
import typer
from pydantic import BaseModel, Field
from typing import Optional
import yaml
from ...utils.factory import PipelineFactory, NodeModelFactory, PipelineBase, DataFactory, EdgeModelFactory, NegativeSamplerFactory
from ...utils.base_model import EarlyStopConfig, DeviceEnum

from ...utils.yaml_dump import deep_convert_dict, merge_comment
import ruamel.yaml
from ruamel.yaml.comments import CommentedMap

class EdgepredPipelineCfg(BaseModel):
    hidden_size: int = 256
    node_embed_size: Optional[int] = -1
    early_stop: Optional[EarlyStopConfig] = EarlyStopConfig()
    eval_batch_size: int = 32769
    train_batch_size: int = 32769
    num_epochs: int = 200
    eval_period: int = 5
    optimizer: dict = {"name": "Adam", "lr": 0.005}
    loss: str = "BCELoss"

pipeline_comments = {
    "hidden_size": "The intermediate hidden size between node model and edge model",
    "node_embed_size": "The node learnable embedding size, -1 to disable",
    "eval_batch_size": "Edge batch size when evaluating",
    "train_batch_size": "Edge batch size when training",
    "num_epochs": "Number of training epochs",
    "eval_period": "Interval epochs between evaluations",
    "early_stop": {
        "patience": "Steps before early stop",
        "checkpoint_path": "Early stop checkpoint model file path"
    }
}

# def merge_comment(cfg_dict, comment_dict):



@PipelineFactory.register("edgepred")
class EdgepredPipeline(PipelineBase):

    user_cfg_cls = None
    pipeline_name = "edgepred"

    def __init__(self):
        self.pipeline_name = "edgepred"

    @classmethod
    def setup_user_cfg_cls(cls):
        from ...utils.enter_config import UserConfig

        class EdgePredUserConfig(UserConfig):
            pipeline_name: str = "edgepred"
            data: DataFactory.get_pydantic_config() = Field(..., discriminator="name")
            node_model: NodeModelFactory.get_pydantic_model_config() = Field(...,
                                                                             discriminator="name")
            edge_model: EdgeModelFactory.get_pydantic_model_config() = Field(...,
                                                                             discriminator="name")
            neg_sampler: NegativeSamplerFactory.get_pydantic_model_config() = Field(...,
                                                                                    discriminator="name")
            general_pipeline: EdgepredPipelineCfg = EdgepredPipelineCfg()

        cls.user_cfg_cls = EdgePredUserConfig

    @property
    def user_cfg_cls(self):
        return self.__class__.user_cfg_cls

    def get_cfg_func(self):
        def config(
            data: DataFactory.get_dataset_enum() = typer.Option(..., help="input data name"),
            cfg: str = typer.Option(
                "cfg.yml", help="output configuration path"),
            node_model: NodeModelFactory.get_model_enum() = typer.Option(...,
                                                                         help="Model name"),
            edge_model: EdgeModelFactory.get_model_enum() = typer.Option(...,
                                                                         help="Model name"),
            neg_sampler: NegativeSamplerFactory.get_model_enum() = typer.Option(
                "uniform", help="Negative sampler name"),
            device: DeviceEnum = typer.Option(
                "cpu", help="Device, cpu or cuda"),
        ):
            self.__class__.setup_user_cfg_cls()
            generated_cfg = {
                "pipeline_name": "edgepred",
                "device": device.value,
                "data": {"name": data.name},
                "neg_sampler": {"name": neg_sampler.value},
                "node_model": {"name": node_model.value},
                "edge_model": {"name": edge_model.value},
            }
            output_cfg = self.user_cfg_cls(**generated_cfg).dict()
            output_cfg = deep_convert_dict(output_cfg)
            comment_dict = {
                "general_pipeline": pipeline_comments,
                "node_model": NodeModelFactory.get_constructor_doc_dict(node_model.value),
                "edge_model": EdgeModelFactory.get_constructor_doc_dict(edge_model.value),
                "neg_sampler": NegativeSamplerFactory.get_constructor_doc_dict(neg_sampler.value)
            }
            comment_dict = merge_comment(output_cfg, comment_dict)
            print(comment_dict)
            yaml = ruamel.yaml.YAML()
            yaml.dump(comment_dict, Path(cfg).open("w"))
            print("Configuration file is generated at {}".format(Path(cfg).absolute()))

        return config

    @classmethod
    def gen_script(cls, user_cfg_dict):
        cls.setup_user_cfg_cls()
        # Check validation
        user_cfg = cls.user_cfg_cls(**user_cfg_dict)
        file_current_dir = Path(__file__).resolve().parent
        with open(file_current_dir / "edgepred.jinja-py", "r") as f:
            template = Template(f.read())

        render_cfg = copy.deepcopy(user_cfg_dict)
        render_cfg["node_model_code"] = NodeModelFactory.get_source_code(
            user_cfg_dict["node_model"]["name"])
        render_cfg["edge_model_code"] = EdgeModelFactory.get_source_code(
            user_cfg_dict["edge_model"]["name"])
        render_cfg["node_model_class_name"] = NodeModelFactory.get_model_class_name(
            user_cfg_dict["node_model"]["name"])
        render_cfg["edge_model_class_name"] = EdgeModelFactory.get_model_class_name(
            user_cfg_dict["edge_model"]["name"])
        render_cfg["neg_sampler_name"] = NegativeSamplerFactory.get_model_class_name(
            user_cfg_dict["neg_sampler"]["name"])
        render_cfg["loss"] = user_cfg_dict["general_pipeline"]["loss"]
        render_cfg.update(DataFactory.get_generated_code_dict(user_cfg_dict["data"]["name"], '**cfg["data"]'))
        generated_user_cfg = copy.deepcopy(user_cfg_dict)
        if len(generated_user_cfg["data"]) == 1:
            generated_user_cfg.pop("data")
        else:
            generated_user_cfg["data"].pop("name")
        generated_user_cfg.pop("pipeline_name")
        generated_user_cfg["node_model"].pop("name")
        generated_user_cfg["edge_model"].pop("name")
        generated_user_cfg["neg_sampler"].pop("name")
        generated_user_cfg["general_pipeline"]["optimizer"].pop("name")
        generated_user_cfg["general_pipeline"].pop("loss")
        generated_train_cfg = copy.deepcopy(user_cfg_dict["general_pipeline"])
        generated_train_cfg["optimizer"].pop("name")

        render_cfg["user_cfg_str"] = f"cfg = {str(generated_user_cfg)}"
        render_cfg["user_cfg"] = user_cfg_dict
        return template.render(**render_cfg)

    @staticmethod
    def get_description() -> str:
        return "Edge classification pipeline"
