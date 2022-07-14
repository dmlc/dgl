from ..pipeline import *
from ..utils.factory import ModelFactory, PipelineFactory
import typer
from enum import Enum
import typing
import yaml
from pathlib import Path

config_app = typer.Typer(help="Generate a configuration file")
for key, pipeline in PipelineFactory.registry.items():
    config_app.command(key, help=pipeline.get_description())(pipeline.get_cfg_func())

if __name__ == "__main__":
    config_app()
