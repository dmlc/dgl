import ruamel.yaml
import torch
import typer

from pathlib import Path
from pydantic import Field
from typing import Optional

from ...utils.factory import ApplyPipelineFactory, PipelineBase, DataFactory
from ...utils.yaml_dump import deep_convert_dict, merge_comment

@ApplyPipelineFactory.register("nodepred")
class ApplyNodepredPipeline(PipelineBase):

    def __init__(self):
        self.pipeline_name = "nodepred"

    @classmethod
    def setup_user_cfg_cls(cls):
        from ...utils.enter_config import UserConfig

        cls.user_cfg_cls = UserConfig

    @property
    def user_cfg_cls(self):
        return self.__class__.user_cfg_cls

    def get_cfg_func(self):
        def config(
            data: DataFactory.filter("nodepred").get_dataset_enum() = typer.Option(None, help="input data name"),
            cfg: Optional[str] = typer.Option(None, help="output configuration file path"),
            cpt: str = typer.Option(..., help="input checkpoint file path")
        ):
            cpt_dict = torch.load(cpt)
            # Training configuration
            train_cfg = cpt_dict['cfg']
            if data is None:
                print('data is not specified, use the training dataset')
                data = train_cfg['data_name']
            if cfg is None:
                cfg = "_".join(["apply", "nodepred", data, train_cfg['model_name']]) + ".yaml"

            self.__class__.setup_user_cfg_cls()
            generated_cfg = {
                "pipeline_name": self.pipeline_name,
                "device": train_cfg['device'],
                "data": {"name": data},
                "cpt_path": cpt
            }
            output_cfg = self.user_cfg_cls(**generated_cfg).dict()
            output_cfg = deep_convert_dict(output_cfg)
            comment_dict = {
                "device": "Torch device name, e.g., cpu or cuda or cuda:0",
                "cpt_path": "Path to the checkpoint file"
            }
            comment_dict = merge_comment(output_cfg, comment_dict)

            yaml = ruamel.yaml.YAML()
            yaml.dump(comment_dict, Path(cfg).open("w"))
            print("Configuration file is generated at {}".format(Path(cfg).absolute()))

        return config

    @classmethod
    def gen_script(cls, user_cfg_dict):
        pass

    @staticmethod
    def get_description() -> str:
        return "Node classification pipeline for inference"
