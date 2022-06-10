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

        cls.user_cfg_cls = ApplyNodePredUserConfig

    @property
    def user_cfg_cls(self):
        return self.__class__.user_cfg_cls

    def get_cfg_func(self):
        def config(
        ):
            pass

        return config

    @classmethod
    def gen_script(cls, user_cfg_dict):
        pass

    @staticmethod
    def get_description() -> str:
        return "Node classification neighbor sampling pipeline for inference"
