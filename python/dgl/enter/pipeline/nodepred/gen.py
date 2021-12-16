from pathlib import Path
from jinja2 import Template
import copy
from ...utils.factory import PipelineFactory, ModelFactory
from ...enter_config import UserConfig


def validate_config(user_cfg):
    pass


@PipelineFactory.register("nodepred")
def gen_script(user_cfg_dict):
    user_cfg = UserConfig(**user_cfg_dict)
    # print(user_cfg)
    file_current_dir = Path(__file__).resolve().parent
    with open(file_current_dir / "nodepred.jinja-py", "r") as f:
        template = Template(f.read())
    
    render_cfg = copy.deepcopy(user_cfg_dict)
    print(f"Render cfg: {render_cfg['data']}, {type(render_cfg['data']['name'])}")
    model_code = ModelFactory.get_source_code(user_cfg_dict["model"]["name"])
    render_cfg["model_code"] = model_code

    generated_user_cfg = copy.deepcopy(user_cfg_dict)
    generated_user_cfg.pop("data")
    generated_user_cfg.pop("pipeline_name")
    generated_user_cfg["model"].pop("name")
    generated_user_cfg["general_pipeline"]["optimizer"].pop("name")
    # generated_user_cfg.pop("general_pipeline")
    
    generated_train_cfg = copy.deepcopy(user_cfg_dict["general_pipeline"])
    generated_train_cfg["optimizer"].pop("name")

    render_cfg["user_cfg_str"] = f"cfg = {str(generated_user_cfg)}"
    # render_cfg["user_train_cfg_str"] = f"train_cfg = {str(generated_train_cfg)}"
    render_cfg["user_cfg"] = user_cfg_dict
    with open("output.py", "w") as f:
        return template.render(**render_cfg)