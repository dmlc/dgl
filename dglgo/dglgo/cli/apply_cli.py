from ..utils.factory import ApplyPipelineFactory
import autopep8
import isort
import typer
import yaml

from pathlib import Path

def apply(
    cfg: str = typer.Option(..., help="config yaml file name")
):
    user_cfg = yaml.safe_load(Path(cfg).open("r"))
    pipeline_name = user_cfg["pipeline"]["name"]
    output_file_content = ApplyPipelineFactory.registry[pipeline_name].gen_script(user_cfg)

    f_code = autopep8.fix_code(output_file_content, options={'aggressive': 1})
    f_code = isort.code(f_code)
    code = compile(f_code, 'dglgo_tmp.py', 'exec')
    exec(code,  {'__name__': '__main__'})
