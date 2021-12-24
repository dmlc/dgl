import enum
import logging
from typing import Callable, Dict, Literal, Union
from pathlib import Path
from abc import ABC, abstractmethod, abstractstaticmethod
from .base_model import DGLBaseModel
import yaml
import inspect
from pydantic import create_model_from_typeddict, create_model, Field
from ...data import CoraGraphDataset, CiteseerGraphDataset, RedditDataset

logger = logging.getLogger(__name__)


class PipelineBase(ABC):

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def get_cfg_func(self):
        pass

    @abstractstaticmethod
    def gen_script(user_cfg_dict: dict):
        pass

    @abstractstaticmethod
    def get_description() -> str:
        pass


class DataFactory:
    registry = {}

    @classmethod
    def register(cls, name: str) -> Callable:

        def inner_wrapper(wrapped_class) -> Callable:
            if name in cls.registry:
                logger.warning(
                    'Executor %s already exists. Will replace it', name)
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper
    
    @classmethod
    def get_dataset_enum(cls):
        enum_class = enum.Enum(
            "DatasetName", {v.__name__: k for k, v in cls.registry.items()})
        return enum_class
    
    @classmethod
    def get_dataset_classname(cls, name):
        return cls.registry[name].__name__
    
    @classmethod
    def get_pydantic_config(cls):
        type_annotation_dict = {}
        dataset_list = []
        for k, v in cls.registry.items():
            dataset_name = v.__name__
            class Base(DGLBaseModel):
                name: Literal[dataset_name]
            
            dataset_list.append(create_model(f'{dataset_name}Config', **type_annotation_dict, __base__=Base))
        
        output = dataset_list[0]
        for d in dataset_list[1:]:
            output = Union[output, d]
        return output

DataFactory.register("cora")(CoraGraphDataset)
DataFactory.register("citeseer")(CiteseerGraphDataset)
DataFactory.register("reddit")(RedditDataset)

class PipelineFactory:
    """ The factory class for creating executors"""

    registry: Dict[str, PipelineBase] = {}
    default_config_registry = {}
    """ Internal registry for available executors """

    @classmethod
    def register(cls, name: str) -> Callable:

        def inner_wrapper(wrapped_class) -> Callable:
            if name in cls.registry:
                logger.warning(
                    'Executor %s already exists. Will replace it', name)
            cls.registry[name] = wrapped_class()
            return wrapped_class

        return inner_wrapper
    

    @classmethod
    def register_default_config_generator(cls, name: str) -> Callable:

        def inner_wrapper(wrapped_class) -> Callable:
            if name in cls.registry:
                logger.warning(
                    'Executor %s already exists. Will replace it', name)
            cls.default_config_registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper
    

    @classmethod
    def call_default_config_generator(cls, generator_name, model_name, dataset_name):
        return cls.default_config_registry[generator_name](model_name, dataset_name)


    @classmethod
    def call_generator(cls, generator_name, cfg):
        return cls.registry[generator_name](cfg)

    @classmethod
    def get_pipeline_enum(cls):
        enum_class = enum.Enum(
            "PipelineName", {k: k for k, v in cls.registry.items()})
        return enum_class


model_dir = Path(__file__).parent.parent / "model"

class ModelFactory:
    """ The factory class for creating executors"""

    registry = {}
    code_registry = {}
    """ Internal registry for available executors """

    @classmethod
    def get_model_enum(cls):
        enum_class = enum.Enum(
            "ModelName", {k: k for k, v in cls.registry.items()})
        return enum_class

    @classmethod
    def register(cls, model_name: str, filename) -> Callable:

        def inner_wrapper(wrapped_class) -> Callable:
            if model_name in cls.registry:
                logger.warning(
                    'Executor %s already exists. Will replace it', model_name)
            cls.registry[model_name] = wrapped_class
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
    def get_pydantic_constructor_arg_type(cls, model_name: str):
        model_enum = cls.get_model_enum()
        arg_dict = cls.get_constructor_default_args(model_name)
        type_annotation_dict = {}
        # type_annotation_dict["name"] = Literal[""]     
        exempt_keys = ['self', 'in_size', 'out_size']
        for k, param in arg_dict.items():
            if k not in exempt_keys:
                type_annotation_dict[k] = arg_dict[k]
        class Base(DGLBaseModel):
            name: Literal[model_name]
        return create_model(f'{model_name.upper()}ModelConfig', **type_annotation_dict, __base__=Base)
    
    @classmethod
    def get_pydantic_model_config(cls):
        model_list = []
        for k in cls.registry:
            model_list.append(cls.get_pydantic_constructor_arg_type(k))
        output = model_list[0]
        for m in model_list[1:]:
            output = Union[output, m]
        return output



    @classmethod
    def get_model_class_name(cls, model_name):
        return cls.registry[model_name].__name__

    @classmethod
    def get_constructor_arg_type(cls, model_name):
        sigs = inspect.signature(cls.registry[model_name].__init__)
        type_annotation_dict = {}
        for k, param in dict(sigs.parameters).items():
            type_annotation_dict[k] = param.annotation
        return type_annotation_dict
