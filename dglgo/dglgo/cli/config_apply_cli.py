from ..apply_pipeline import *
import typer

from ..utils.factory import ApplyPipelineFactory

config_apply_app = typer.Typer(
    help="Generate a configuration file for inference"
)
for key, pipeline in ApplyPipelineFactory.registry.items():
    config_apply_app.command(key, help=pipeline.get_description())(
        pipeline.get_cfg_func()
    )
