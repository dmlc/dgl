from ...utils.factory import GraphModelFactory
from .gin_ogbg import OGBGGIN
from .pna import PNA

GraphModelFactory.register("gin")(OGBGGIN)
GraphModelFactory.register("pna")(PNA)
