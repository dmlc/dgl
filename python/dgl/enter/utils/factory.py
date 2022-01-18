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
from ...dataloading.negative_sampler import GlobalUniform, PerSourceUniform
import inspect
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
    import_code_registry = {}

    @classmethod
    def register(cls, name: str, args=(), import_code="import dgl") -> Callable:

        def inner_wrapper(wrapped_class) -> Callable:
            if name in cls.registry:
                logger.warning(
                    'Executor %s already exists. Will replace it', name)
            cls.registry[name] = wrapped_class
            cls.import_code_registry[name] = import_code
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

            dataset_list.append(create_model(
                f'{dataset_name}Config', **type_annotation_dict, __base__=Base))

        output = dataset_list[0]
        for d in dataset_list[1:]:
            output = Union[output, d]
        return output


DataFactory.register("cora", import_code="from dgl.data import CoraGraphDataset")(CoraGraphDataset)
DataFactory.register("citeseer", import_code="from dgl.data import CiteseerGraphDataset")(CiteseerGraphDataset)
DataFactory.register("ogbl-collab", import_code="from ogb.linkproppred import DglLinkPropPredDataset")("DglLinkPropPredDataset('ogbl-collab')")
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

    def __init__(self):
        self.registry = {}
        self.code_registry = {}
    """ Internal registry for available executors """

    def get_model_enum(self):
        enum_class = enum.Enum(
            "ModelName", {k: k for k, v in self.registry.items()})
        return enum_class

    def register(self, model_name: str) -> Callable:

        def inner_wrapper(wrapped_class) -> Callable:
            if model_name in self.registry:
                logger.warning(
                    'Executor %s already exists. Will replace it', model_name)
            self.registry[model_name] = wrapped_class
            # code_filename = model_dir / filename
            code_filename = Path(inspect.getfile(wrapped_class))
            self.code_registry[model_name] = code_filename.read_text()
            return wrapped_class

        return inner_wrapper

    def get_source_code(self, model_name):
        return self.code_registry[model_name]

    def get_constructor_default_args(self, model_name):
        sigs = inspect.signature(self.registry[model_name].__init__)
        default_map = {}
        for k, param in dict(sigs.parameters).items():
            default_map[k] = param.default
        return default_map

    def get_pydantic_constructor_arg_type(self, model_name: str):
        model_enum = self.get_model_enum()
        arg_dict = self.get_constructor_default_args(model_name)
        type_annotation_dict = {}
        # type_annotation_dict["name"] = Literal[""]
        exempt_keys = ['self', 'in_size', 'out_size']
        for k, param in arg_dict.items():
            if k not in exempt_keys:
                type_annotation_dict[k] = arg_dict[k]

        class Base(DGLBaseModel):
            name: Literal[model_name]
        return create_model(f'{model_name.upper()}ModelConfig', **type_annotation_dict, __base__=Base)

    def get_pydantic_model_config(self):
        model_list = []
        for k in self.registry:
            model_list.append(self.get_pydantic_constructor_arg_type(k))
        output = model_list[0]
        for m in model_list[1:]:
            output = Union[output, m]
        return output

    def get_model_class_name(self, model_name):
        return self.registry[model_name].__name__

    def get_constructor_arg_type(self, model_name):
        sigs = inspect.signature(self.registry[model_name].__init__)
        type_annotation_dict = {}
        for k, param in dict(sigs.parameters).items():
            type_annotation_dict[k] = param.annotation
        return type_annotation_dict



class SamplerFactory:
    """ The factory class for creating executors"""

    def __init__(self):
        self.registry = {}

    def get_model_enum(self):
        enum_class = enum.Enum(
            "NegativeSamplerName", {k: k for k, v in self.registry.items()})
        return enum_class

    def register(self, sampler_name: str) -> Callable:

        def inner_wrapper(wrapped_class) -> Callable:
            if sampler_name in self.registry:
                logger.warning(
                    'Sampler %s already exists. Will replace it', sampler_name)
            self.registry[sampler_name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    def get_constructor_default_args(self, sampler_name):
        sigs = inspect.signature(self.registry[sampler_name].__init__)
        default_map = {}
        for k, param in dict(sigs.parameters).items():
            default_map[k] = param.default
        return default_map

    def get_pydantic_constructor_arg_type(self, sampler_name: str):
        model_enum = self.get_model_enum()
        arg_dict = self.get_constructor_default_args(sampler_name)
        type_annotation_dict = {}
        # type_annotation_dict["name"] = Literal[""]
        exempt_keys = ['self', 'in_size', 'out_size', 'redundancy']
        for k, param in arg_dict.items():
            if k not in exempt_keys or param is None:
                if k == 'k' or k == 'redundancy':
                    type_annotation_dict[k] = 3
                else:
                    type_annotation_dict[k] = arg_dict[k]

        class Base(DGLBaseModel):
            name: Literal[sampler_name]
        return create_model(f'{sampler_name.upper()}SamplerConfig', **type_annotation_dict, __base__=Base)

    def get_pydantic_model_config(self):
        model_list = []
        for k in self.registry:
            model_list.append(self.get_pydantic_constructor_arg_type(k))
        output = model_list[0]
        for m in model_list[1:]:
            output = Union[output, m]
        return output

    def get_model_class_name(self, model_name):
        return self.registry[model_name].__name__

    def get_constructor_arg_type(self, model_name):
        sigs = inspect.signature(self.registry[model_name].__init__)
        type_annotation_dict = {}
        for k, param in dict(sigs.parameters).items():
            type_annotation_dict[k] = param.annotation
        return type_annotation_dict



NegativeSamplerFactory = SamplerFactory()
NegativeSamplerFactory.register("uniform")(GlobalUniform)
NegativeSamplerFactory.register("persource")(PerSourceUniform)

NodeModelFactory = ModelFactory()
EdgeModelFactory = ModelFactory()
