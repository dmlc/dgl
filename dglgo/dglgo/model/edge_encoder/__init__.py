from ...utils.factory import EdgeModelFactory
from .ele import ElementWiseProductPredictor
from .bilinear import BilinearPredictor

EdgeModelFactory.register("ele")(ElementWiseProductPredictor)
EdgeModelFactory.register("bilinear")(BilinearPredictor)