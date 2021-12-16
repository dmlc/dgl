from ..utils.factory import ModelFactory
from ..enter_config import PipelineEnum, UserConfig
import typer
from enum import Enum
import typing
import yaml
from pathlib import Path


def config(
    model: ModelFactory.enum_class = typer.Option(..., help="Model name"),
    output: str = typer.Option(..., help="output config name"),
    data: str = typer.Option(..., help="input data name"),
    pipeline: PipelineEnum = typer.Option(..., help="Pipeline name"),
):  
    cfg = {}
    cfg["pipeline_name"] = pipeline.value
    cfg["data"] = {"name": data}
    model_config = ModelFactory.get_constructor_default_args(model.value)
    model_config.pop("self")
    model_config.pop("in_size")
    model_config.pop("out_size")
    cfg["model"] = {
        "name": model.value,
        **model_config
    }
    print(cfg)
    output_cfg = UserConfig(**cfg).dict()
    print(output_cfg)
    yaml.safe_dump(output_cfg, Path(output).open("w"), sort_keys=False)

if __name__ == "__main__":
    config_app = typer.Typer()
    config_app.command()(config)
    config_app()