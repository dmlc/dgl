from ...utils.factory import EdgeModelFactory
from .bilinear import BilinearPredictor
from .ele import ElementWiseProductPredictor

EdgeModelFactory.register("ele")(ElementWiseProductPredictor)
EdgeModelFactory.register("bilinear")(BilinearPredictor)
