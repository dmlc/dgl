from ..pipeline import *
from ..utils.factory import ModelFactory, PipelineFactory
from ..enter_config import UserConfig, output_file_path
import typer
from enum import Enum
import typing
import yaml
from pathlib import Path


# def closure()


config_app = typer.Typer(help="Please specify your pipeline at the COMMAND")
# PipelineFactory.initilize_app(config_app)
for key, pipeline in PipelineFactory.registry.items():
    # print(pipeline.app)
    config_app.command(key, help=pipeline.get_description())(pipeline.get_cfg_func())

# @config_app.callback(help="callback")
# def callback(output_file: Path =  typer.Option("cfg.yaml", help="Output config yaml file path")):
#     output_file_path = output_file

    # config_app.add_typer(pipeline.app, name=key, help="pipeline name")
    # break
# def config(
#     model: ModelFactory.get_model_enum() = typer.Option(..., help="Model name"),
#     output: str = typer.Option(..., help="output config name"),
#     data: str = typer.Option(..., help="input data name"),
#     pipeline: PipelineFactory.get_pipeline_enum() = typer.Option(..., help="Pipeline name"),
# ):  
#     cfg = {}
#     cfg["pipeline_name"] = pipeline.value
#     cfg["data"] = {"name": data}
#     model_config = ModelFactory.get_constructor_default_args(model.value)
#     model_config.pop("self")
#     model_config.pop("in_size")
#     model_config.pop("out_size")
#     cfg["model"] = {
#         "name": model.value,
#         **model_config
#     }
#     cfg["general_pipeline"] = PipelineFactory.call_default_config_generator(pipeline, model, data)
#     print(cfg)
#     output_cfg = UserConfig(**cfg).dict()
#     print(output_cfg)
#     yaml.safe_dump(output_cfg, Path(output).open("w"), sort_keys=False)

if __name__ == "__main__":
    # config_app = typer.Typer()
    # for key, pipeline in PipelineFactory.registry.items():
    #     print(pipeline.app)
    #     config_app.add_typer(pipeline.app, name=key)
    # # config_app.command()(config)
    config_app()