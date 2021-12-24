from pathlib import Path
from jinja2 import Template
import copy
import typer
from pydantic import BaseModel, Field
from typing import Optional
import yaml
from ...utils.factory import PipelineFactory, ModelFactory, PipelineBase, DataFactory
from ...utils.base_model import EarlyStopConfig

class NodepredPipelineCfg(BaseModel):
    node_embed_size: Optional[int] = -1
    early_stop: Optional[EarlyStopConfig] = EarlyStopConfig()
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
            data: DataFactory.get_dataset_enum() = typer.Option(..., help="input data name"),
            cfg: str = typer.Option(
                "cfg.yml", help="output configuration path"),
            model: ModelFactory.get_model_enum() = typer.Option(..., help="Model name"),
            device: str = typer.Option("cpu", help="Device, cpu or cuda"),
        ):
            from ...utils.enter_config import UserConfig
            generated_cfg = {
                "pipeline_name": "nodepred",
                "device": device,
                "data": {"name": data.name},
                "model": {"name": model.value},
                "general_pipeline": NodepredPipelineCfg()
            }
            output_cfg = UserConfig(**generated_cfg).dict()
            yaml.safe_dump(output_cfg, Path(cfg).open("w"), sort_keys=False)

        return config

    @staticmethod
    def gen_script(user_cfg_dict):
        from ...utils.enter_config import UserConfig
        # Check validation
        user_cfg = UserConfig(**user_cfg_dict)
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

        generated_train_cfg = copy.deepcopy(user_cfg_dict["general_pipeline"])
        generated_train_cfg["optimizer"].pop("name")

        render_cfg["user_cfg_str"] = f"cfg = {str(generated_user_cfg)}"
        render_cfg["user_cfg"] = user_cfg_dict
        return template.render(**render_cfg)

    @staticmethod
    def get_description() -> str:
        return "Node classification pipeline"