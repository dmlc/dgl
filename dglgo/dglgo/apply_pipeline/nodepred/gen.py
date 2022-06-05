import typer
from typing import Optional
from ...utils.factory import ApplyPipelineFactory, PipelineBase, DataFactory

@ApplyPipelineFactory.register("nodepred")
class ApplyNodepredPipeline(PipelineBase):

    def __init__(self):
        super().__init__()

    def get_cfg_func(self):
        def config(
            data: DataFactory.filter("nodepred").get_dataset_enum() = typer.Option(None, help="input data name"),
            cfg: Optional[str] = typer.Option(None, help="output configuration file path"),
            cpt: str = typer.Option(..., help="input checkpoint file path")
        ):
            # Set default values for data and cfg if not specified
            pass

        return config

    @classmethod
    def gen_script(cls, user_cfg_dict):
        pass

    @staticmethod
    def get_description() -> str:
        return "Node classification pipeline for inference"
