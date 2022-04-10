from pathlib import Path
from jinja2 import Template
import copy
import typer
from pydantic import BaseModel
from typing import Optional
from ...utils.factory import PipelineFactory, PipelineBase
from ...utils.yaml_dump import deep_convert_dict, merge_comment
import ruamel.yaml

pipeline_comments = {
    "num_runs": "Number of experiments to run",
}

class GraphpredPipelineCfg(BaseModel):
    num_runs: int = 1

@PipelineFactory.register("graphpred")
class GraphpredPipeline(PipelineBase):
    def __init__(self):
        self.pipeline_name = "graphpred"

    @classmethod
    def setup_user_cfg_cls(cls):
        from ...utils.enter_config import UserConfig
        class GraphPredUserConfig(UserConfig):
            general_pipeline: GraphpredPipelineCfg = GraphpredPipelineCfg()

        cls.user_cfg_cls = GraphPredUserConfig

    @property
    def user_cfg_cls(self):
        return self.__class__.user_cfg_cls

    def get_cfg_func(self):
        def config(
            cfg: Optional[str] = typer.Option(
                None, help="output configuration path"),
        ):
            self.__class__.setup_user_cfg_cls()
            generated_cfg = {
                "pipeline_name": self.pipeline_name,
                "device": "cpu",
                "general_pipeline": {}
            }
            output_cfg = self.user_cfg_cls(**generated_cfg).dict()
            output_cfg = deep_convert_dict(output_cfg)
            comment_dict = {
                "device": "Torch device name, e.q. cpu or cuda or cuda:0",
                "general_pipeline": pipeline_comments,
            }
            comment_dict = merge_comment(output_cfg, comment_dict)

            yaml = ruamel.yaml.YAML()
            yaml.dump(comment_dict, Path(cfg).open("w"))
            print("Configuration file is generated at {}".format(Path(cfg).absolute()))

        return config

    @classmethod
    def gen_script(cls, user_cfg_dict):
        cls.setup_user_cfg_cls()
        file_current_dir = Path(__file__).resolve().parent
        with open(file_current_dir / "graphpred.jinja-py", "r") as f:
            template = Template(f.read())

        render_cfg = copy.deepcopy(user_cfg_dict)

        generated_user_cfg = copy.deepcopy(user_cfg_dict)
        generated_user_cfg.pop("pipeline_name")

        generated_train_cfg = copy.deepcopy(user_cfg_dict["general_pipeline"])

        render_cfg["user_cfg_str"] = f"cfg = {str(generated_user_cfg)}"
        render_cfg["user_cfg"] = user_cfg_dict
        return template.render(**render_cfg)

    @staticmethod
    def get_description() -> str:
        return "Graph property prediction pipeline"
