import enum
import inspect
import logging
from abc import ABC, abstractmethod, abstractstaticmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import yaml
from dgl.dataloading.negative_sampler import GlobalUniform, PerSourceUniform
from numpydoc import docscrape
from pydantic import create_model, create_model_from_typeddict, Field
from typing_extensions import Literal

from .base_model import DGLBaseModel

logger = logging.getLogger(__name__)

ALL_PIPELINE = ["nodepred", "nodepred-ns", "linkpred", "graphpred"]


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


class DataFactoryClass:
    def __init__(self):
        self.registry = {}
        self.pipeline_name = None
        self.pipeline_allowed = {}

    def register(
        self,
        name: str,
        import_code: str,
        class_name: str,
        allowed_pipeline: List[str],
        extra_args={},
    ):
        self.registry[name] = {
            "name": name,
            "import_code": import_code,
            "class_name": class_name,
            "extra_args": extra_args,
        }
        for pipeline in allowed_pipeline:
            if pipeline in self.pipeline_allowed:
                self.pipeline_allowed[pipeline].append(name)
            else:
                self.pipeline_allowed[pipeline] = [name]
        return self

    def get_dataset_enum(self):
        enum_class = enum.Enum(
            "DatasetName", {v["name"]: k for k, v in self.registry.items()}
        )
        return enum_class

    def get_dataset_classname(self, name):
        return self.registry[name]["class_name"]

    def get_constructor_arg_type(self, model_name):
        sigs = inspect.signature(self.registry[model_name].__init__)
        type_annotation_dict = {}
        for k, param in dict(sigs.parameters).items():
            type_annotation_dict[k] = param.annotation
        return type_annotation_dict

    def get_pydantic_config(self):

        type_annotation_dict = {}
        dataset_list = []
        for k, v in self.registry.items():
            dataset_name = v["name"]
            type_annotation_dict = v["extra_args"]
            if "name" in type_annotation_dict:
                del type_annotation_dict["name"]
            base = self.get_base_class(dataset_name, self.pipeline_name)
            dataset_list.append(
                create_model(
                    f"{dataset_name}Config",
                    **type_annotation_dict,
                    __base__=base,
                )
            )

        output = dataset_list[0]
        for d in dataset_list[1:]:
            output = Union[output, d]
        return output

    def get_import_code(self, name):
        return self.registry[name]["import_code"]

    def get_import_code(self, name):
        return self.registry[name]["import_code"]

    def get_extra_args(self, name):
        return self.registry[name]["extra_args"]

    def get_class_name(self, name):
        return self.registry[name]["class_name"]

    def get_generated_code_dict(self, name, args='**cfg["data"]'):
        d = {}
        d["data_import_code"] = self.registry[name]["import_code"]
        data_initialize_code = self.registry[name]["class_name"]
        extra_args_dict = self.registry[name]["extra_args"]
        if len(extra_args_dict) > 0:
            data_initialize_code = data_initialize_code.format('**cfg["data"]')
        d["data_initialize_code"] = data_initialize_code
        return d

    def filter(self, pipeline_name):
        allowed_name = self.pipeline_allowed[pipeline_name]
        new_registry = {
            k: v for k, v in self.registry.items() if k in allowed_name
        }
        d = DataFactoryClass()
        d.registry = new_registry
        d.pipeline_name = pipeline_name
        return d

    @staticmethod
    def get_base_class(dataset_name, pipeline_name):
        if pipeline_name == "linkpred":

            class EdgeBase(DGLBaseModel):
                name: Literal[dataset_name]
                split_ratio: Optional[Tuple[float, float, float]] = None
                neg_ratio: Optional[int] = None

            return EdgeBase
        else:

            class NodeBase(DGLBaseModel):
                name: Literal[dataset_name]
                split_ratio: Optional[Tuple[float, float, float]] = None

            return NodeBase


DataFactory = DataFactoryClass()

DataFactory.register(
    "cora",
    import_code="from dgl.data import CoraGraphDataset",
    class_name="CoraGraphDataset()",
    allowed_pipeline=["nodepred", "nodepred-ns", "linkpred"],
)

DataFactory.register(
    "citeseer",
    import_code="from dgl.data import CiteseerGraphDataset",
    class_name="CiteseerGraphDataset()",
    allowed_pipeline=["nodepred", "nodepred-ns", "linkpred"],
)

DataFactory.register(
    "pubmed",
    import_code="from dgl.data import PubmedGraphDataset",
    class_name="PubmedGraphDataset()",
    allowed_pipeline=["nodepred", "nodepred-ns", "linkpred"],
)

DataFactory.register(
    "csv",
    import_code="from dgl.data import CSVDataset",
    extra_args={"data_path": "./"},
    class_name="CSVDataset({})",
    allowed_pipeline=["nodepred", "nodepred-ns", "linkpred", "graphpred"],
)

DataFactory.register(
    "reddit",
    import_code="from dgl.data import RedditDataset",
    class_name="RedditDataset()",
    allowed_pipeline=["nodepred", "nodepred-ns", "linkpred"],
)

DataFactory.register(
    "co-buy-computer",
    import_code="from dgl.data import AmazonCoBuyComputerDataset",
    class_name="AmazonCoBuyComputerDataset()",
    allowed_pipeline=["nodepred", "nodepred-ns", "linkpred"],
)

DataFactory.register(
    "ogbn-arxiv",
    import_code="from ogb.nodeproppred import DglNodePropPredDataset",
    extra_args={},
    class_name="DglNodePropPredDataset('ogbn-arxiv')",
    allowed_pipeline=["nodepred", "nodepred-ns", "linkpred"],
)

DataFactory.register(
    "ogbn-products",
    import_code="from ogb.nodeproppred import DglNodePropPredDataset",
    extra_args={},
    class_name="DglNodePropPredDataset('ogbn-products')",
    allowed_pipeline=["nodepred", "nodepred-ns", "linkpred"],
)

DataFactory.register(
    "ogbl-collab",
    import_code="from ogb.linkproppred import DglLinkPropPredDataset",
    extra_args={},
    class_name="DglLinkPropPredDataset('ogbl-collab')",
    allowed_pipeline=["linkpred"],
)

DataFactory.register(
    "ogbl-citation2",
    import_code="from ogb.linkproppred import DglLinkPropPredDataset",
    extra_args={},
    class_name="DglLinkPropPredDataset('ogbl-citation2')",
    allowed_pipeline=["linkpred"],
)

DataFactory.register(
    "ogbg-molhiv",
    import_code="from ogb.graphproppred import DglGraphPropPredDataset",
    extra_args={},
    class_name="DglGraphPropPredDataset(name='ogbg-molhiv')",
    allowed_pipeline=["graphpred"],
)

DataFactory.register(
    "ogbg-molpcba",
    import_code="from ogb.graphproppred import DglGraphPropPredDataset",
    extra_args={},
    class_name="DglGraphPropPredDataset(name='ogbg-molpcba')",
    allowed_pipeline=["graphpred"],
)


class PipelineFactory:
    """The factory class for creating executors"""

    registry: Dict[str, PipelineBase] = {}
    default_config_registry = {}
    """ Internal registry for available executors """

    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_class) -> Callable:
            if name in cls.registry:
                logger.warning(
                    "Executor %s already exists. Will replace it", name
                )
            cls.registry[name] = wrapped_class()
            return wrapped_class

        return inner_wrapper

    @classmethod
    def register_default_config_generator(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_class) -> Callable:
            if name in cls.registry:
                logger.warning(
                    "Executor %s already exists. Will replace it", name
                )
            cls.default_config_registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def call_default_config_generator(
        cls, generator_name, model_name, dataset_name
    ):
        return cls.default_config_registry[generator_name](
            model_name, dataset_name
        )

    @classmethod
    def call_generator(cls, generator_name, cfg):
        return cls.registry[generator_name](cfg)

    @classmethod
    def get_pipeline_enum(cls):
        enum_class = enum.Enum(
            "PipelineName", {k: k for k, v in cls.registry.items()}
        )
        return enum_class


class ApplyPipelineFactory:
    """The factory class for creating executors for inference"""

    registry: Dict[str, PipelineBase] = {}
    """ Internal registry for available executors """

    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_class) -> Callable:
            if name in cls.registry:
                logger.warning(
                    "Executor %s already exists. Will replace it", name
                )
            cls.registry[name] = wrapped_class()
            return wrapped_class

        return inner_wrapper


model_dir = Path(__file__).parent.parent / "model"


class ModelFactory:
    """The factory class for creating executors"""

    def __init__(self):
        self.registry = {}
        self.code_registry = {}

    """ Internal registry for available executors """

    def get_model_enum(self):
        enum_class = enum.Enum(
            "ModelName", {k: k for k, v in self.registry.items()}
        )
        return enum_class

    def register(self, model_name: str) -> Callable:
        def inner_wrapper(wrapped_class) -> Callable:
            if model_name in self.registry:
                logger.warning(
                    "Executor %s already exists. Will replace it", model_name
                )
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
        exempt_keys = ["self", "in_size", "out_size", "data_info"]
        for k, param in arg_dict.items():
            if k not in exempt_keys:
                type_annotation_dict[k] = arg_dict[k]

        class Base(DGLBaseModel):
            name: Literal[model_name]

        return create_model(
            f"{model_name.upper()}ModelConfig",
            **type_annotation_dict,
            __base__=Base,
        )

    def get_constructor_doc_dict(self, name):
        model_class = self.registry[name]
        docs = inspect.getdoc(model_class.__init__)
        param_docs = docscrape.NumpyDocString(docs)
        param_docs_dict = {}
        for param in param_docs["Parameters"]:
            param_docs_dict[param.name] = param.desc[0]
        return param_docs_dict

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

    def filter(self, filter_func):
        new_fac = ModelFactory()
        for name in self.registry:
            if filter_func(self.registry[name]):
                new_fac.registry[name] = self.registry[name]
                new_fac.code_registry[name] = self.code_registry[name]
        return new_fac


class SamplerFactory:
    """The factory class for creating executors"""

    def __init__(self):
        self.registry = {}

    def get_model_enum(self):
        enum_class = enum.Enum(
            "NegativeSamplerName", {k: k for k, v in self.registry.items()}
        )
        return enum_class

    def register(self, sampler_name: str) -> Callable:
        def inner_wrapper(wrapped_class) -> Callable:
            if sampler_name in self.registry:
                logger.warning(
                    "Sampler %s already exists. Will replace it", sampler_name
                )
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
        exempt_keys = ["self", "in_size", "out_size", "redundancy"]
        for k, param in arg_dict.items():
            if k not in exempt_keys or param is None:
                if k == "k" or k == "redundancy":
                    type_annotation_dict[k] = 3
                else:
                    type_annotation_dict[k] = arg_dict[k]

        class Base(DGLBaseModel):
            name: Literal[sampler_name]

        return create_model(
            f"{sampler_name.upper()}SamplerConfig",
            **type_annotation_dict,
            __base__=Base,
        )

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

    def get_constructor_doc_dict(self, name):
        model_class = self.registry[name]
        docs = inspect.getdoc(model_class)
        param_docs = docscrape.NumpyDocString(docs)
        param_docs_dict = {}
        for param in param_docs["Parameters"]:
            param_docs_dict[param.name] = param.desc[0]
        return param_docs_dict


NegativeSamplerFactory = SamplerFactory()
NegativeSamplerFactory.register("global")(GlobalUniform)
NegativeSamplerFactory.register("persource")(PerSourceUniform)

NodeModelFactory = ModelFactory()
EdgeModelFactory = ModelFactory()
GraphModelFactory = ModelFactory()
