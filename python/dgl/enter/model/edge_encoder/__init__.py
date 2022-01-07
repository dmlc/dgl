from ...utils.factory import EdgeModelFactory
from .edgepred import EdgePredictor
from .transe import TransE

EdgeModelFactory.register("edgepred")(EdgePredictor)
EdgeModelFactory.register("transe")(TransE)