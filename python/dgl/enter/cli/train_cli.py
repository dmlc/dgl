from ..utils.factory import ModelFactory
from ..enter_config import PipelineEnum, UserConfig
import typer
from enum import Enum
import typing
import yaml
from pathlib import Path


# @config_app.command()
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
def train(
    cfg: str = typer.Option(..., help="yaml file name"),
    export: bool = typer.Option(False, "--export"),
    run: bool = typer.Option(True, help = "Whether to execute the code"),
    output: str = typer.Option("output.py", help="output file name")
):
    user_cfg = yaml.safe_load(Path(cfg).open("r"))
    pipeline_name = user_cfg["pipeline_name"]
    print(f"cfg: {user_cfg['data']}")
    output_file_content = PipelineFactory.call_generator(
        pipeline_name, user_cfg)

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

# def train(
#     code_file: str = typer.Argument("output.py", help="output config name"),
# ):  
#     import importlib.util
#     spec = importlib.util.spec_from_file_location("main", code_file)
#     main_module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(main_module)
#     main_module.main()


# if __name__ == "__main__":
#     train_app = typer.Typer()
#     train_app.command()(train)
#     train_app()