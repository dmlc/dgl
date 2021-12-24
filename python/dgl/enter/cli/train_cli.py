from ..utils.factory import ModelFactory, PipelineFactory
from ..utils.enter_config import UserConfig
import typer
from enum import Enum
import typing
import yaml
from pathlib import Path

import isort
import autopep8

def train(
    cfg: str = typer.Option(..., help="yaml file name"),
    export: bool = typer.Option(False, "--export"),
    run: bool = typer.Option(True, help = "Whether to execute the code"),
    output: str = typer.Option("output.py", help="output file name")
):
    user_cfg = yaml.safe_load(Path(cfg).open("r"))
    pipeline_name = user_cfg["pipeline_name"]
    output_file_content = PipelineFactory.registry[pipeline_name].gen_script(user_cfg)

    f_code = autopep8.fix_code(output_file_content, options={'aggressive': 1})
    f_code = isort.code(f_code)
    if export:
        with open(output, "w") as f:
            f.write(f_code)
    if run:
        exec(f_code,  {'__name__': '__main__'})


if __name__ == "__main__":
    train_app = typer.Typer()
    train_app.command()(train)
    train_app()