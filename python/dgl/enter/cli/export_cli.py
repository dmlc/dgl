from ..utils.factory import ModelFactory, PipelineFactory
from ..enter_config import PipelineEnum, UserConfig
import typer
from enum import Enum
import typing
import yaml
from pathlib import Path

import isort
import autopep8

# @config_app.command()
def export(
    yaml_filename: str = typer.Option(..., help="yaml file name"),
    output: str = typer.Option("output.py", help="output file name")
):
    user_cfg = yaml.safe_load(Path(yaml_filename).open("r"))
    pipeline_name = user_cfg["pipeline_name"]
    print(f"cfg: {user_cfg['data']}")
    output_file_content = PipelineFactory.call_generator(
        pipeline_name, user_cfg)
    with open(output, "w") as f:
        f_code = autopep8.fix_code(output_file_content, options={'aggressive': 1})
        f_code = isort.code(f_code)
        f.write(f_code)
    typer.echo("Done")



if __name__ == "__main__":
    export_app = typer.Typer()
    export_app.command()(export)
    export_app()
