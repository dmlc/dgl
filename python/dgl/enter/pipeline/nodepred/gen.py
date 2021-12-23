from pathlib import Path
from jinja2 import Template
import copy
import typer
from pydantic import BaseModel, Field
from typing import Optional
import yaml
from ...utils.factory import PipelineFactory, ModelFactory, PipelineBase


class NodepredPipelineCfg(BaseModel):
    node_embed_size: Optional[int] = -1
    early_stop: Optional[dict]
    num_epochs: int = 200
    eval_period: int = 5
    optimizer: dict = {"name": "Adam", "lr": 0.005}
    loss: str = "CrossEntropyLoss"


@PipelineFactory.register("nodepred")
class NodepredPipeline(PipelineBase):
    def __init__(self):
        self.pipeline_name = "nodepred"

    def get_cfg_func(self):
        def config(
                data: str = typer.Option(..., help="input data name"),
                cfg: str = typer.Option("cfg.yml", help="output configuration path"),
                model: ModelFactory.get_model_enum() = typer.Option(..., help="Model name"),):
            from ...enter_config import UserConfig
            self.default_cfg = NodepredPipelineCfg()
            generated_cfg = {}
            generated_cfg["pipeline_name"] = "nodepred"
            generated_cfg["data"] = {"name": data}
            model_config = ModelFactory.get_constructor_default_args(model.value)
            model_config = ModelFactory.get_pydantic_constructor_arg_type(model.value)
            import ipdb
            ipdb.set_trace()
            model_config.pop("self")
            model_config.pop("in_size")
            model_config.pop("out_size")
            generated_cfg["model"] = {
                "name": model.value,
                **model_config
            }
            generated_cfg["general_pipeline"] = NodepredPipelineCfg()
            print(generated_cfg)
            output_cfg = UserConfig(**generated_cfg).dict()
            print(output_cfg)
            yaml.safe_dump(output_cfg, Path(cfg).open("w"), sort_keys=False)

        return config

    @staticmethod
    def gen_script(user_cfg_dict):
        from ...enter_config import UserConfig
        user_cfg = UserConfig(**user_cfg_dict)
        # print(user_cfg)
        file_current_dir = Path(__file__).resolve().parent
        with open(file_current_dir / "nodepred.jinja-py", "r") as f:
            template = Template(f.read())

        render_cfg = copy.deepcopy(user_cfg_dict)
        model_code = ModelFactory.get_source_code(
            user_cfg_dict["model"]["name"])
        render_cfg["model_code"] = model_code
        render_cfg["model_class_name"] = ModelFactory.get_model_class_name(
            user_cfg_dict["model"]["name"])

        generated_user_cfg = copy.deepcopy(user_cfg_dict)
        generated_user_cfg.pop("data")
        generated_user_cfg.pop("pipeline_name")
        generated_user_cfg["model"].pop("name")
        generated_user_cfg["general_pipeline"]["optimizer"].pop("name")
        # generated_user_cfg.pop("general_pipeline")

        generated_train_cfg = copy.deepcopy(user_cfg_dict["general_pipeline"])
        generated_train_cfg["optimizer"].pop("name")

        render_cfg["user_cfg_str"] = f"cfg = {str(generated_user_cfg)}"
        # render_cfg["user_train_cfg_str"] = f"train_cfg = {str(generated_train_cfg)}"
        render_cfg["user_cfg"] = user_cfg_dict
        return template.render(**render_cfg)
        # with open("output.py", "w") as f:
        #     return template.render(**render_cfg)

    @staticmethod
    def get_description() -> str:
        return "Node classification pipeline"
# @PipelineFactory.register("nodepred")
# def gen_script(user_cfg_dict):
#     from ...enter_config import UserConfig
#     user_cfg = UserConfig(**user_cfg_dict)
#     # print(user_cfg)
#     file_current_dir = Path(__file__).resolve().parent
#     with open(file_current_dir / "nodepred.jinja-py", "r") as f:
#         template = Template(f.read())

#     render_cfg = copy.deepcopy(user_cfg_dict)
#     model_code = ModelFactory.get_source_code(user_cfg_dict["model"]["name"])
#     render_cfg["model_code"] = model_code
#     render_cfg["model_class_name"] = ModelFactory.get_model_class_name(user_cfg_dict["model"]["name"])

#     generated_user_cfg = copy.deepcopy(user_cfg_dict)
#     generated_user_cfg.pop("data")
#     generated_user_cfg.pop("pipeline_name")
#     generated_user_cfg["model"].pop("name")
#     generated_user_cfg["general_pipeline"]["optimizer"].pop("name")
#     # generated_user_cfg.pop("general_pipeline")

#     generated_train_cfg = copy.deepcopy(user_cfg_dict["general_pipeline"])
#     generated_train_cfg["optimizer"].pop("name")

#     render_cfg["user_cfg_str"] = f"cfg = {str(generated_user_cfg)}"
#     # render_cfg["user_train_cfg_str"] = f"train_cfg = {str(generated_train_cfg)}"
#     render_cfg["user_cfg"] = user_cfg_dict
#     with open("output.py", "w") as f:
#         return template.render(**render_cfg)
