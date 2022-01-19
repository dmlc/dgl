from ...utils.factory import EdgeModelFactory
from .ele import ElementWiseProductPredictor
from .bilinear import BilinearPredictor
from .transe import TransE

EdgeModelFactory.register("ele")(ElementWiseProductPredictor)
EdgeModelFactory.register("bilinear")(BilinearPredictor)
# EdgeModelFactory.register("transe")(TransE)