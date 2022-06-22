from ...utils.factory import ApplyPipelineFactory, PipelineBase

@ApplyPipelineFactory.register("graphpred")
class ApplyGraphpredPipeline(PipelineBase):
    def __init__(self):
        self.pipeline = {
            "name": "graphpred",
            "mode": "apply"
        }
