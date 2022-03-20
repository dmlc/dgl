from ..utils.factory import ModelFactory, PipelineFactory
from ..utils.enter_config import UserConfig
import typer
from enum import Enum
import typing
import yaml
from pathlib import Path

import isort
import autopep8

def export(
    cfg: str = typer.Option("cfg.yaml", help="config yaml file name"),
    output: str = typer.Option("script.py", help="output python file name")
):
    user_cfg = yaml.safe_load(Path(cfg).open("r"))
    pipeline_name = user_cfg["pipeline_name"]
    output_file_content = PipelineFactory.registry[pipeline_name].gen_script(user_cfg)

    f_code = autopep8.fix_code(output_file_content, options={'aggressive': 1})
    f_code = isort.code(f_code)
    with open(output, "w") as f:
        f.write(f_code)
    print("The python script is generated at {}, based on config file {}".format(Path(output).absolute(), Path(cfg).absolute()))

if __name__ == "__main__":
    export_app = typer.Typer()
    export_app.command()(export)
    export_app()