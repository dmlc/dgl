#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import os
import shutil
import sys
import sysconfig

from setuptools import find_packages, setup
from setuptools.dist import Distribution
from setuptools.extension import Extension


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


CURRENT_DIR = os.path.dirname(__file__)


def get_lib_path():
    """Get library path, name and version"""
    # We can not import `libinfo.py` in setup.py directly since __init__.py
    # Will be invoked which introduces dependences
    libinfo_py = os.path.join(CURRENT_DIR, "./dgl/_ffi/libinfo.py")
    libinfo = {"__file__": libinfo_py}
    exec(
        compile(open(libinfo_py, "rb").read(), libinfo_py, "exec"),
        libinfo,
        libinfo,
    )
    version = libinfo["__version__"]

    lib_path = libinfo["find_lib_path"]()
    libs = [lib_path[0]]

    return libs, version


def get_lib_pattern(lib_name):
    if sys.platform.startswith("linux"):
        lib_pattern = f"lib{lib_name}_*.so"
    elif sys.platform.startswith("darwin"):
        lib_pattern = f"lib{lib_name}_*.dylib"
    elif sys.platform.startswith("win"):
        lib_pattern = f"{lib_name}_*.dll"
    else:
        raise NotImplementedError("Unsupported system: %s" % sys.platform)
    return lib_pattern


LIBS, VERSION = get_lib_path()
BACKENDS = ["pytorch"]


def remove_lib(lib_name):
    for lib_path in glob.glob(
        os.path.join(CURRENT_DIR, "dgl", lib_name, get_lib_pattern(lib_name))
    ):
        try:
            os.remove(lib_path)
        except BaseException:
            pass


def cleanup():
    # Wheel cleanup
    try:
        os.remove("MANIFEST.in")
    except BaseException:
        pass

    for path in LIBS:
        _, libname = os.path.split(path)
        try:
            os.remove(os.path.join("dgl", libname))
        except BaseException:
            pass
    for backend in BACKENDS:
        remove_lib("tensoradapter")

        if backend == "pytorch":
            remove_lib("dgl_sparse")
            remove_lib("graphbolt")

    # Remove build artifacts.
    dir_to_remove = ["build", "dgl.egg-info"]
    for dir_ in dir_to_remove:
        print(f"Removing {dir_}")
        if os.path.isdir(dir_):
            shutil.rmtree(dir_)


def config_cython():
    """Try to configure cython and return cython configuration"""
    if sys.platform.startswith("win"):
        print(
            "WARNING: Cython is not supported on Windows, will compile without cython module"
        )
        return []
    sys_cflags = sysconfig.get_config_var("CFLAGS")

    if "i386" in sys_cflags and "x86_64" in sys_cflags:
        print(
            "WARNING: Cython library may not be compiled correctly with both i386 and x64"
        )
        return []
    try:
        from Cython.Build import cythonize

        # from setuptools.extension import Extension
        if sys.version_info >= (3, 0):
            subdir = "_cy3"
        else:
            subdir = "_cy2"
        ret = []
        path = "dgl/_ffi/_cython"
        library_dirs = ["dgl", "../build/Release", "../build"]
        libraries = ["dgl"]
        for fn in os.listdir(path):
            if not fn.endswith(".pyx"):
                continue
            ret.append(
                Extension(
                    "dgl._ffi.%s.%s" % (subdir, fn[:-4]),
                    ["dgl/_ffi/_cython/%s" % fn],
                    include_dirs=[
                        "../include/",
                        "../third_party/dmlc-core/include",
                        "../third_party/dlpack/include",
                    ],
                    library_dirs=library_dirs,
                    libraries=libraries,
                    # Crashes without this flag with GCC 5.3.1
                    extra_compile_args=["-std=c++17"],
                    language="c++",
                )
            )
        return cythonize(
            ret, force=True, compiler_directives={"language_level": "3"}
        )
    except ImportError:
        print(
            "WARNING: Cython is not installed, will compile without cython module"
        )
        return []


def copy_lib(lib_name, backend=""):
    for lib_path in glob.glob(
        os.path.join(dir_, lib_name, backend, get_lib_pattern(lib_name))
    ):
        lib_file_name = os.path.basename(lib_path)
        dst_dir_ = os.path.join(CURRENT_DIR, "dgl", lib_name, backend)
        os.makedirs(
            dst_dir_,
            exist_ok=True,
        )
        shutil.copy(
            os.path.join(dir_, lib_name, backend, lib_file_name),
            dst_dir_,
        )
        fo.write(f"include dgl/{lib_name}/{backend}/{lib_file_name}\n")


include_libs = False
wheel_include_libs = False
if "bdist_wheel" in sys.argv or os.getenv("CONDA_BUILD"):
    wheel_include_libs = True
elif "clean" in sys.argv:
    cleanup()
else:
    include_libs = True

setup_kwargs = {}

# For bdist_wheel only
if wheel_include_libs:
    with open("MANIFEST.in", "w") as fo:
        for path in LIBS:
            shutil.copy(path, os.path.join(CURRENT_DIR, "dgl"))
            dir_, libname = os.path.split(path)
            fo.write("include dgl/%s\n" % libname)

        for backend in BACKENDS:
            copy_lib("tensoradapter", backend)
            if backend == "pytorch":
                copy_lib("dgl_sparse")
                copy_lib("graphbolt")
    setup_kwargs = {"include_package_data": True}


def get_lib_file_path(lib_name, backend=""):
    return (
        f"dgl/{lib_name}/{backend}",
        glob.glob(
            os.path.join(
                os.path.dirname(os.path.relpath(path, CURRENT_DIR)),
                lib_name,
                backend,
                get_lib_pattern(lib_name),
            )
        ),
    )


# For source tree setup
# Conda build also includes the binary library
if include_libs:
    rpath = [os.path.relpath(path, CURRENT_DIR) for path in LIBS]
    data_files = [("dgl", rpath)]
    for path in LIBS:
        for backend in BACKENDS:
            data_files.append(get_lib_file_path("tensoradapter", backend))
            if backend == "pytorch":
                data_files.append(get_lib_file_path("dgl_sparse"))
                data_files.append(get_lib_file_path("graphbolt"))
    setup_kwargs = {"include_package_data": True, "data_files": data_files}

# Configure dependencies.
install_requires = [
    "networkx>=2.1",
    "numpy>=1.14.0",
    "packaging",
    "pandas",
    "psutil>=5.8.0",
    "pydantic>=2.0",
    "pyyaml",
    "requests>=2.19.0",
    "scipy>=1.1.0",
    "tqdm",
]

setup(
    name="dgl" + os.getenv("DGL_PACKAGE_SUFFIX", ""),
    version=VERSION,
    description="Deep Graph Library",
    zip_safe=False,
    maintainer="DGL Team",
    maintainer_email="wmjlyjemaine@gmail.com",
    packages=find_packages(),
    install_requires=install_requires,
    url="https://github.com/dmlc/dgl",
    distclass=BinaryDistribution,
    ext_modules=config_cython(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    license="APACHE",
    **setup_kwargs,
)

if wheel_include_libs:
    cleanup()
