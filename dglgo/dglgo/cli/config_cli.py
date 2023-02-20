from ..pipeline import *
import typing
from enum import Enum
from pathlib import Path

import typer
import yaml

from ..utils.factory import ModelFactory, PipelineFactory

config_app = typer.Typer(help="Generate a configuration file")
for key, pipeline in PipelineFactory.registry.items():
    config_app.command(key, help=pipeline.get_description())(
        pipeline.get_cfg_func()
    )

if __name__ == "__main__":
    config_app()
