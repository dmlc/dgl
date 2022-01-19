import yaml
import jinja2
from jinja2 import Template
from enum import Enum, IntEnum
import copy
from pydantic import BaseModel, ValidationError


# class PipelineEnum(str, Enum):
#     nodepred = "nodepred"
# class UserConfig(BaseModel):
#     pipeline_name: PipelineEnum



def main(cfg_yaml_filename):    
    with open(cfg_yaml_filename) as f:
        user_cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    if user_cfg["pipeline_name"] == "nodepred": 
        render_cfg = {}
        with open("/home/ubuntu/dev/csr/dgl/python/dgl/enter/pipeline/nodepred/nodepred.j2", "r") as f:
            template = Template(f.read())
        print(user_cfg["model"]["name"])
        if user_cfg["model"]["name"] == "GCN":
            with open("model/gcn.py", "r") as f:
                render_cfg["model_code"] = f.read()
        if user_cfg["model"]["name"] == "GAT":
            with open("model/gat.py", "r") as f:
                render_cfg["model_code"] = f.read()

        generated_user_cfg = copy.deepcopy(user_cfg)
        generated_user_cfg.pop("data")
        generated_user_cfg.pop("pipeline_name")
        generated_user_cfg["model"].pop("name")
        generated_user_cfg["optimizer"].pop("name")
        # pop out empty config
        # pop_key = []
        # for k, v in generated_user_cfg.items():
        #     if isinstance(v, dict) and len(v) == 0:
        #         pop_key.append(k)
        # for key in pop_key:
        #     generated_user_cfg.pop(key)
        render_cfg["user_cfg_str"] = f"cfg = {str(generated_user_cfg)}"
        render_cfg["user_cfg"] = user_cfg
        with open("output.py", "w") as f:
            f.write(template.render(**render_cfg))
    
    if user_cfg["pipeline_name"] == "nodepred-ns":         
        render_cfg = {}
        with open("/home/ubuntu/dev/csr/dgl/python/dgl/enter/pipeline/nodepred-sample/nodepred-ns.jinja-py", "r") as f:
            template = Template(f.read())
        print(user_cfg["model"]["name"])
        if user_cfg["model"]["name"] == "GCN":
            with open("model/gcn_ns.py", "r") as f:
                render_cfg["model_code"] = f.read()
        # if user_cfg["model"]["name"] == "GAT":
        #     with open("model/gat.py", "r") as f:
        #         render_cfg["model_code"] = f.read()

        generated_user_cfg = copy.deepcopy(user_cfg)
        generated_user_cfg.pop("data")
        generated_user_cfg.pop("pipeline_name")
        generated_user_cfg["model"].pop("name")
        generated_user_cfg["optimizer"].pop("name")
        # pop out empty config
        # pop_key = []
        # for k, v in generated_user_cfg.items():
        #     if isinstance(v, dict) and len(v) == 0:
        #         pop_key.append(k)
        # for key in pop_key:
        #     generated_user_cfg.pop(key)
        render_cfg["user_cfg_str"] = f"cfg = {str(generated_user_cfg)}"
        render_cfg["user_cfg"] = user_cfg
        with open("output.py", "w") as f:
            f.write(template.render(**render_cfg))
    
main()