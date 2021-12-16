import enum
import logging
from typing import Callable
from pathlib import Path

import inspect
logger = logging.getLogger(__name__)

class PipelineFactory:
    """ The factory class for creating executors"""

    registry = {}
    """ Internal registry for available executors """

    @classmethod
    def register(cls, name: str) -> Callable:

        def inner_wrapper(wrapped_class) -> Callable:
            if name in cls.registry:
                logger.warning('Executor %s already exists. Will replace it', name)
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper
    
    @classmethod
    def call_generator(cls, generator_name, cfg):
        return cls.registry[generator_name](cfg)

model_dir = Path(__file__).parent.parent / "model"

class ModelFactory:
    """ The factory class for creating executors"""

    registry = {}
    code_registry = {}
    enum_class = None
    """ Internal registry for available executors """

    @classmethod
    def register(cls, model_name: str, filename) -> Callable:

        def inner_wrapper(wrapped_class) -> Callable:
            if model_name in cls.registry:
                logger.warning('Executor %s already exists. Will replace it', model_name)
            cls.registry[model_name] = wrapped_class
            cls.enum_class = enum.Enum("ModelName", {k: k for k,v in cls.registry.items()})
            code_filename = model_dir / filename
            cls.code_registry[model_name] = code_filename.read_text()
            return wrapped_class

        return inner_wrapper
    
    @classmethod
    def get_source_code(cls, model_name):
        return cls.code_registry[model_name]
    
    @classmethod
    def get_constructor_default_args(cls, model_name):
        sigs = inspect.signature(cls.registry[model_name].__init__)
        default_map = {}
        for k, param in dict(sigs.parameters).items():
            default_map[k] = param.default
        return default_map
    

    @classmethod
    def get_constructor_arg_type(cls, model_name):
        sigs = inspect.signature(cls.registry[model_name].__init__)
        type_annotation_dict = {}
        for k, param in dict(sigs.parameters).items():
            type_annotation_dict[k] = param.annotation
        return type_annotation_dict