import typer
from typing import Optional
from ...utils.factory import ApplyPipelineFactory, PipelineBase, DataFactory

@ApplyPipelineFactory.register("nodepred")
class ApplyNodepredPipeline(PipelineBase):

    def get_cfg_func(self):
        def config(
            data: DataFactory.filter("nodepred").get_dataset_enum() = typer.Option(..., help="input data name"),
            cfg: Optional[str] = typer.Option(None, help="output configuration file path"),
            cpt: str = typer.Option(..., help="input checkpoint file path")
        ):
            pass

        return config

    @staticmethod
    def get_description() -> str:
        return "Node classification pipeline for inference"
