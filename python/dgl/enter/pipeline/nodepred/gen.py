from pathlib import Path
from jinja2 import Template
import copy

def gen_script(user_cfg, output_filename="output.py"):
    current_dir = Path(__file__).resolve().parent
    with open(current_dir / "nodepred.jinja-py", "r") as f:
        template = Template(f.read())
    
    render_cfg = copy.deepcopy(user_cfg)
    if user_cfg["model"]["name"] == "GCN":
            with open("model/gcn.py", "r") as f:
                render_cfg["model_code"] = f.read()    

    generated_user_cfg = copy.deepcopy(user_cfg)
    generated_user_cfg.pop("data")
    generated_user_cfg.pop("pipeline_name")
    generated_user_cfg["model"].pop("name")
    generated_user_cfg["optimizer"].pop("name")

    render_cfg["user_cfg_str"] = f"cfg = {str(generated_user_cfg)}"
    render_cfg["user_cfg"] = user_cfg
    with open("output.py", "w") as f:
        f.write(template.render(**render_cfg))