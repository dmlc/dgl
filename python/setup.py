#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import os
import platform
import shutil
import sys
import sysconfig

from setuptools import find_packages
from setuptools.dist import Distribution

# need to use distutils.core for correct placement of cython dll
if "--inplace" in sys.argv:
    from distutils.core import setup
    from distutils.extension import Extension
else:
    from setuptools import setup
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


def get_ta_lib_pattern():
    if sys.platform.startswith("linux"):
        ta_lib_pattern = "libtensoradapter_*.so"
    elif sys.platform.startswith("darwin"):
        ta_lib_pattern = "libtensoradapter_*.dylib"
    elif sys.platform.startswith("win"):
        ta_lib_pattern = "tensoradapter_*.dll"
    else:
        raise NotImplementedError("Unsupported system: %s" % sys.platform)
    return ta_lib_pattern


def get_dgl_sparse_pattern():
    if sys.platform.startswith("linux"):
        dgl_sparse_lib_pattern = "libdgl_sparse_*.so"
    elif sys.platform.startswith("darwin"):
        dgl_sparse_lib_pattern = "libdgl_sparse_*.dylib"
    elif sys.platform.startswith("win"):
        dgl_sparse_lib_pattern = "dgl_sparse_*.dll"
    else:
        raise NotImplementedError("Unsupported system: %s" % sys.platform)
    return dgl_sparse_lib_pattern


LIBS, VERSION = get_lib_path()
BACKENDS = ["pytorch"]
TA_LIB_PATTERN = get_ta_lib_pattern()
SPARSE_LIB_PATTERN = get_dgl_sparse_pattern()


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
        for ta_path in glob.glob(
            os.path.join(
                CURRENT_DIR, "dgl", "tensoradapter", backend, TA_LIB_PATTERN
            )
        ):
            try:
                os.remove(ta_path)
            except BaseException:
                pass

        if backend == "pytorch":
            for sparse_path in glob.glob(
                os.path.join(
                    CURRENT_DIR, "dgl", "dgl_sparse", SPARSE_LIB_PATTERN
                )
            ):
                try:
                    os.remove(sparse_path)
                except BaseException:
                    pass


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
                    extra_compile_args=["-std=c++11"],
                    language="c++",
                )
            )
        return cythonize(ret, force=True)
    except ImportError:
        print(
            "WARNING: Cython is not installed, will compile without cython module"
        )
        return []


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
            for ta_path in glob.glob(
                os.path.join(dir_, "tensoradapter", backend, TA_LIB_PATTERN)
            ):
                ta_name = os.path.basename(ta_path)
                os.makedirs(
                    os.path.join(CURRENT_DIR, "dgl", "tensoradapter", backend),
                    exist_ok=True,
                )
                shutil.copy(
                    os.path.join(dir_, "tensoradapter", backend, ta_name),
                    os.path.join(CURRENT_DIR, "dgl", "tensoradapter", backend),
                )
                fo.write(
                    "include dgl/tensoradapter/%s/%s\n" % (backend, ta_name)
                )
            if backend == 'pytorch':
                for sparse_path in glob.glob(
                    os.path.join(dir_, "dgl_sparse", SPARSE_LIB_PATTERN)
                ):
                    sparse_name = os.path.basename(sparse_path)
                    os.makedirs(
                        os.path.join(CURRENT_DIR, "dgl", "dgl_sparse"),
                        exist_ok=True,
                    )
                    shutil.copy(
                        os.path.join(dir_, "dgl_sparse", sparse_name),
                        os.path.join(CURRENT_DIR, "dgl", "dgl_sparse"),
                    )
                    fo.write(
                        "include dgl/dgl_sparse/%s\n" % sparse_name
                    )


    setup_kwargs = {"include_package_data": True}

# For source tree setup
# Conda build also includes the binary library
if include_libs:
    rpath = [os.path.relpath(path, CURRENT_DIR) for path in LIBS]
    data_files = [("dgl", rpath)]
    for path in LIBS:
        for backend in BACKENDS:
            data_files.append(
                (
                    "dgl/tensoradapter/%s" % backend,
                    glob.glob(
                        os.path.join(
                            os.path.dirname(os.path.relpath(path, CURRENT_DIR)),
                            "tensoradapter",
                            backend,
                            TA_LIB_PATTERN,
                        )
                    ),
                )
            )
            if backend == 'pytorch':
                data_files.append(
                    (
                        "dgl/dgl_sparse",
                        glob.glob(
                            os.path.join(
                                os.path.dirname(os.path.relpath(path, CURRENT_DIR)),
                                "dgl_sparse",
                                SPARSE_LIB_PATTERN,
                            )
                        ),
                    )
                )
    setup_kwargs = {"include_package_data": True, "data_files": data_files}

setup(
    name="dgl" + os.getenv("DGL_PACKAGE_SUFFIX", ""),
    version=VERSION,
    description="Deep Graph Library",
    zip_safe=False,
    maintainer="DGL Team",
    maintainer_email="wmjlyjemaine@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.14.0",
        "scipy>=1.1.0",
        "networkx>=2.1",
        "requests>=2.19.0",
        "tqdm",
        "psutil>=5.8.0",
    ],
    url="https://github.com/dmlc/dgl",
    distclass=BinaryDistribution,
    ext_modules=config_cython(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    license="APACHE",
    **setup_kwargs
)

if wheel_include_libs:
    cleanup()
