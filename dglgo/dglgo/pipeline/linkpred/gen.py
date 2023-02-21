import copy
from pathlib import Path
from typing import Optional

import ruamel.yaml
import typer
import yaml
from jinja2 import Template
from pydantic import BaseModel, Field
from ruamel.yaml.comments import CommentedMap

from ...utils.base_model import DeviceEnum, EarlyStopConfig
from ...utils.factory import (
    DataFactory,
    EdgeModelFactory,
    NegativeSamplerFactory,
    NodeModelFactory,
    PipelineBase,
    PipelineFactory,
)

from ...utils.yaml_dump import deep_convert_dict, merge_comment


class LinkpredPipelineCfg(BaseModel):
    hidden_size: int = 256
    eval_batch_size: int = 32769
    train_batch_size: int = 32769
    num_epochs: int = 200
    eval_period: int = 5
    optimizer: dict = {"name": "Adam", "lr": 0.005}
    loss: str = "BCELoss"
    save_path: str = "results"
    num_runs: int = 1


pipeline_comments = {
    "hidden_size": "The intermediate hidden size between node model and edge model",
    "eval_batch_size": "Edge batch size when evaluating",
    "train_batch_size": "Edge batch size when training",
    "num_epochs": "Number of training epochs",
    "eval_period": "Interval epochs between evaluations",
    "save_path": "Directory to save the experiment results",
    "num_runs": "Number of experiments to run",
}


@PipelineFactory.register("linkpred")
class LinkpredPipeline(PipelineBase):

    user_cfg_cls = None
    pipeline_name = "linkpred"

    def __init__(self):
        self.pipeline = {"name": "linkpred", "mode": "train"}

    @classmethod
    def setup_user_cfg_cls(cls):
        from ...utils.enter_config import UserConfig

        class LinkPredUserConfig(UserConfig):
            data: DataFactory.filter("linkpred").get_pydantic_config() = Field(
                ..., discriminator="name"
            )
            node_model: NodeModelFactory.get_pydantic_model_config() = Field(
                ..., discriminator="name"
            )
            edge_model: EdgeModelFactory.get_pydantic_model_config() = Field(
                ..., discriminator="name"
            )
            neg_sampler: NegativeSamplerFactory.get_pydantic_model_config() = (
                Field(..., discriminator="name")
            )
            general_pipeline: LinkpredPipelineCfg = LinkpredPipelineCfg()

        cls.user_cfg_cls = LinkPredUserConfig

    @property
    def user_cfg_cls(self):
        return self.__class__.user_cfg_cls

    def get_cfg_func(self):
        def config(
            data: DataFactory.filter(
                "linkpred"
            ).get_dataset_enum() = typer.Option(..., help="input data name"),
            cfg: str = typer.Option(
                "cfg.yaml", help="output configuration path"
            ),
            node_model: NodeModelFactory.get_model_enum() = typer.Option(
                ..., help="Model name"
            ),
            edge_model: EdgeModelFactory.get_model_enum() = typer.Option(
                ..., help="Model name"
            ),
            neg_sampler: NegativeSamplerFactory.get_model_enum() = typer.Option(
                "persource", help="Negative sampler name"
            ),
        ):
            self.__class__.setup_user_cfg_cls()
            generated_cfg = {
                "pipeline_name": self.pipeline["name"],
                "pipeline_mode": self.pipeline["mode"],
                "device": "cpu",
                "data": {"name": data.name},
                "neg_sampler": {"name": neg_sampler.value},
                "node_model": {"name": node_model.value},
                "edge_model": {"name": edge_model.value},
            }
            output_cfg = self.user_cfg_cls(**generated_cfg).dict()
            output_cfg = deep_convert_dict(output_cfg)
            comment_dict = {
                "device": "Torch device name, e.g., cpu or cuda or cuda:0",
                "general_pipeline": pipeline_comments,
                "node_model": NodeModelFactory.get_constructor_doc_dict(
                    node_model.value
                ),
                "edge_model": EdgeModelFactory.get_constructor_doc_dict(
                    edge_model.value
                ),
                "neg_sampler": NegativeSamplerFactory.get_constructor_doc_dict(
                    neg_sampler.value
                ),
                "data": {
                    "split_ratio": "List of float, e.q. [0.8, 0.1, 0.1]. Split ratios for training, validation and test sets. Must sum to one. Leave blank to use builtin split in original dataset",
                    "neg_ratio": "Int, e.q. 2. Indicate how much negative samples to be sampled per positive samples. Leave blank to use builtin split in original dataset",
                },
            }
            comment_dict = merge_comment(output_cfg, comment_dict)

            if cfg is None:
                cfg = (
                    "_".join(
                        [
                            "linkpred",
                            data.value,
                            node_model.value,
                            edge_model.value,
                        ]
                    )
                    + ".yaml"
                )
            yaml = ruamel.yaml.YAML()
            yaml.dump(comment_dict, Path(cfg).open("w"))
            print(
                "Configuration file is generated at {}".format(
                    Path(cfg).absolute()
                )
            )

        return config

    @classmethod
    def gen_script(cls, user_cfg_dict):
        cls.setup_user_cfg_cls()
        # Check validation
        user_cfg = cls.user_cfg_cls(**user_cfg_dict)
        file_current_dir = Path(__file__).resolve().parent
        with open(file_current_dir / "linkpred.jinja-py", "r") as f:
            template = Template(f.read())

        render_cfg = copy.deepcopy(user_cfg_dict)
        render_cfg["node_model_code"] = NodeModelFactory.get_source_code(
            user_cfg_dict["node_model"]["name"]
        )
        render_cfg["edge_model_code"] = EdgeModelFactory.get_source_code(
            user_cfg_dict["edge_model"]["name"]
        )
        render_cfg[
            "node_model_class_name"
        ] = NodeModelFactory.get_model_class_name(
            user_cfg_dict["node_model"]["name"]
        )
        render_cfg[
            "edge_model_class_name"
        ] = EdgeModelFactory.get_model_class_name(
            user_cfg_dict["edge_model"]["name"]
        )
        render_cfg[
            "neg_sampler_name"
        ] = NegativeSamplerFactory.get_model_class_name(
            user_cfg_dict["neg_sampler"]["name"]
        )
        render_cfg["loss"] = user_cfg_dict["general_pipeline"]["loss"]
        # update import and initialization code
        render_cfg.update(
            DataFactory.get_generated_code_dict(
                user_cfg_dict["data"]["name"], '**cfg["data"]'
            )
        )
        generated_user_cfg = copy.deepcopy(user_cfg_dict)
        if len(generated_user_cfg["data"]) == 1:
            generated_user_cfg.pop("data")
        else:
            generated_user_cfg["data"].pop("name")
        generated_user_cfg.pop("pipeline_name")
        generated_user_cfg.pop("pipeline_mode")
        generated_user_cfg["node_model"].pop("name")
        generated_user_cfg["edge_model"].pop("name")
        generated_user_cfg["neg_sampler"].pop("name")
        generated_user_cfg["general_pipeline"]["optimizer"].pop("name")
        generated_user_cfg["general_pipeline"].pop("loss")
        generated_train_cfg = copy.deepcopy(user_cfg_dict["general_pipeline"])
        generated_train_cfg["optimizer"].pop("name")

        if user_cfg_dict["data"].get("split_ratio", None) is not None:
            assert (
                user_cfg_dict["data"].get("neg_ratio", None) is not None
            ), "Please specify both split_ratio and neg_ratio"
            render_cfg[
                "data_initialize_code"
            ] = "{}, split_ratio={}, neg_ratio={}".format(
                render_cfg["data_initialize_code"],
                user_cfg_dict["data"]["split_ratio"],
                user_cfg_dict["data"]["neg_ratio"],
            )
            generated_user_cfg["data"].pop("split_ratio")
            generated_user_cfg["data"].pop("neg_ratio")

        render_cfg["user_cfg_str"] = f"cfg = {str(generated_user_cfg)}"
        render_cfg["user_cfg"] = user_cfg_dict
        return template.render(**render_cfg)

    @staticmethod
    def get_description() -> str:
        return "Link prediction pipeline"
