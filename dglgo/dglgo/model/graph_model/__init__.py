from ...utils.factory import GraphModelFactory
from .gin_ogbg import OGBGGIN

GraphModelFactory.register("gin")(OGBGGIN)