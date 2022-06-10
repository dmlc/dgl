from pydantic import Field

from ...utils.factory import ApplyPipelineFactory, PipelineBase, DataFactory

@ApplyPipelineFactory.register("nodepred-ns")
class ApplyNodepredNsPipeline(PipelineBase):

    def __init__(self):
        self.pipeline = {
            "name": "nodepred-ns",
            "mode": "apply"
        }

    @classmethod
    def setup_user_cfg_cls(cls):
        from ...utils.enter_config import UserConfig
        class ApplyNodePredUserConfig(UserConfig):
            data: DataFactory.filter("nodepred").get_pydantic_config() = Field(..., discriminator="name")
