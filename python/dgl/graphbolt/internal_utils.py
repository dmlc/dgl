"""Miscallenous internal utils."""
import functools
import hashlib
import os
import platform
import warnings
from collections.abc import Mapping, Sequence

import requests
import torch
from tqdm.auto import tqdm

try:
    from packaging import version  # pylint: disable=unused-import
except ImportError:
    # If packaging isn't installed, try and use the vendored copy in setuptools
    from setuptools.extern.packaging import version


@functools.lru_cache(maxsize=None)
def is_wsl(v: str = platform.uname().release) -> int:
    """Detects if Python is running in WSL"""

    if v.endswith("-Microsoft"):
        return 1
    elif v.endswith("microsoft-standard-WSL2"):
        return 2

    return 0


# pylint: disable=invalid-name
_default_formatwarning = warnings.formatwarning


def built_with_cuda():
    """Returns whether GraphBolt was built with CUDA support."""
    # This op is defined if graphbolt is built with CUDA support.
    return hasattr(torch.ops.graphbolt, "set_max_uva_threads")


class GBWarning(UserWarning):
    """GraphBolt Warning class."""


# pylint: disable=unused-argument
def gb_warning_format(message, category, filename, lineno, line=None):
    """Format GraphBolt warnings."""
    if isinstance(category, GBWarning):
        return "GraphBolt Warning: {}\n".format(message)
    else:
        return _default_formatwarning(
            message, category, filename, lineno, line=None
        )


def gb_warning(message, category=GBWarning, stacklevel=2):
    """GraphBolt warning wrapper that defaults to ``GBWarning`` instead of
    ``UserWarning`` category.
    """
    return warnings.warn(message, category=category, stacklevel=stacklevel)


warnings.formatwarning = gb_warning_format


def is_listlike(data):
    """Return if the data is a sequence but not a string."""
    return isinstance(data, Sequence) and not isinstance(data, str)


def recursive_apply(data, fn, *args, **kwargs):
    """Recursively apply a function to every element in a container.

    If the input data is a list or any sequence other than a string, returns a list
    whose elements are the same elements applied with the given function.

    If the input data is a dict or any mapping, returns a dict whose keys are the same
    and values are the elements applied with the given function.

    If the input data is a nested container, the result will have the same nested
    structure where each element is transformed recursively.

    The first argument of the function will be passed with the individual elements from
    the input data, followed by the arguments in :attr:`args` and :attr:`kwargs`.

    Parameters
    ----------
    data : any
        Any object.
    fn : callable
        Any function.
    args, kwargs :
        Additional arguments and keyword-arguments passed to the function.

    Examples
    --------
    Applying a ReLU function to a dictionary of tensors:

    >>> h = {k: torch.randn(3) for k in ['A', 'B', 'C']}
    >>> h = recursive_apply(h, torch.nn.functional.relu)
    >>> assert all((v >= 0).all() for v in h.values())
    """
    if isinstance(data, Mapping):
        return {
            k: recursive_apply(v, fn, *args, **kwargs) for k, v in data.items()
        }
    elif isinstance(data, tuple):
        return tuple(recursive_apply(v, fn, *args, **kwargs) for v in data)
    elif is_listlike(data):
        return [recursive_apply(v, fn, *args, **kwargs) for v in data]
    else:
        return fn(data, *args, **kwargs)


def recursive_apply_reduce_all(data, fn, *args, **kwargs):
    """Recursively apply a function to every element in a container and reduce
    the boolean results with all.

    If the input data is a list or any sequence other than a string, returns
    True if and only if the given function returns True for all elements.

    If the input data is a dict or any mapping, returns True if and only if the
    given function returns True for values.

    If the input data is a nested container, the result will be reduced over the
    nested structure where each element is tested recursively.

    The first argument of the function will be passed with the individual elements from
    the input data, followed by the arguments in :attr:`args` and :attr:`kwargs`.

    Parameters
    ----------
    data : any
        Any object.
    fn : callable
        Any function returning a boolean.
    args, kwargs :
        Additional arguments and keyword-arguments passed to the function.
    """
    if isinstance(data, Mapping):
        return all(
            recursive_apply_reduce_all(v, fn, *args, **kwargs)
            for v in data.values()
        )
    elif isinstance(data, tuple) or is_listlike(data):
        return all(
            recursive_apply_reduce_all(v, fn, *args, **kwargs) for v in data
        )
    else:
        return fn(data, *args, **kwargs)


def get_nonproperty_attributes(_obj) -> list:
    """Get attributes of the class except for the properties."""
    attributes = [
        attribute
        for attribute in dir(_obj)
        if not attribute.startswith("__")
        and (
            not hasattr(type(_obj), attribute)
            or not isinstance(getattr(type(_obj), attribute), property)
        )
        and not callable(getattr(_obj, attribute))
    ]
    return attributes


def get_attributes(_obj) -> list:
    """Get attributes of the class."""
    attributes = [
        attribute
        for attribute in dir(_obj)
        if not attribute.startswith("__")
        and not callable(getattr(_obj, attribute))
    ]
    return attributes


def download(
    url,
    path=None,
    overwrite=True,
    sha1_hash=None,
    retries=5,
    verify_ssl=True,
    log=True,
):
    """Download a given URL.

    Codes borrowed from mxnet/gluon/utils.py

    Parameters
    ----------
    url : str
        URL to download.
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with the same name as in url.
    overwrite : bool, optional
        Whether to overwrite the destination file if it already exists.
        By default always overwrites the downloaded file.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    retries : integer, default 5
        The number of times to attempt downloading in case of failure or non 200 return codes.
    verify_ssl : bool, default True
        Verify SSL certificates.
    log : bool, default True
        Whether to print the progress for download

    Returns
    -------
    str
        The file path of the downloaded file.
    """
    if path is None:
        fname = url.split("/")[-1]
        # Empty filenames are invalid
        assert fname, (
            "Can't construct file-name from this URL. "
            "Please set the `path` option manually."
        )
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split("/")[-1])
        else:
            fname = path
    assert retries >= 0, "Number of retries should be at least 0"

    if not verify_ssl:
        warnings.warn(
            "Unverified HTTPS request is being made (verify_ssl=False). "
            "Adding certificate verification is strongly advised."
        )

    if (
        overwrite
        or not os.path.exists(fname)
        or (sha1_hash and not check_sha1(fname, sha1_hash))
    ):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        while retries + 1 > 0:
            # Disable pyling too broad Exception
            # pylint: disable=W0703
            try:
                if log:
                    print("Downloading %s from %s..." % (fname, url))
                r = requests.get(url, stream=True, verify=verify_ssl)
                if r.status_code != 200:
                    raise RuntimeError("Failed downloading url %s" % url)
                # Get the total file size.
                total_size = int(r.headers.get("content-length", 0))
                with tqdm(
                    total=total_size, unit="B", unit_scale=True, desc=fname
                ) as progress_bar:
                    with open(fname, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024):
                            if chunk:  # filter out keep-alive new chunks
                                f.write(chunk)
                                progress_bar.update(len(chunk))
                if sha1_hash and not check_sha1(fname, sha1_hash):
                    raise UserWarning(
                        "File {} is downloaded but the content hash does not match."
                        " The repo may be outdated or download may be incomplete. "
                        'If the "repo_url" is overridden, consider switching to '
                        "the default repo.".format(fname)
                    )
                break
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    raise e
                if log:
                    print(
                        "download failed, retrying, {} attempt{} left".format(
                            retries, "s" if retries > 1 else ""
                        )
                    )

    return fname


def check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.

    Codes borrowed from mxnet/gluon/utils.py

    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.

    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, "rb") as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash


def extract_archive(file, target_dir, overwrite=True):
    """Extract archive file.

    Parameters
    ----------
    file : str
        Absolute path of the archive file.
    target_dir : str
        Target directory of the archive to be uncompressed.
    overwrite : bool, default True
        Whether to overwrite the contents inside the directory.
        By default always overwrites.
    """
    if os.path.exists(target_dir) and not overwrite:
        return
    print("Extracting file to {}".format(target_dir))
    if (
        file.endswith(".tar.gz")
        or file.endswith(".tar")
        or file.endswith(".tgz")
    ):
        import tarfile

        with tarfile.open(file, "r") as archive:

            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                prefix = os.path.commonprefix([abs_directory, abs_target])
                return prefix == abs_directory

            def safe_extract(
                tar, path=".", members=None, *, numeric_owner=False
            ):
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
                tar.extractall(path, members, numeric_owner=numeric_owner)

            safe_extract(archive, path=target_dir)
    elif file.endswith(".gz"):
        import gzip
        import shutil

        with gzip.open(file, "rb") as f_in:
            target_file = os.path.join(target_dir, os.path.basename(file)[:-3])
            with open(target_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    elif file.endswith(".zip"):
        import zipfile

        with zipfile.ZipFile(file, "r") as archive:
            archive.extractall(path=target_dir)
    else:
        raise Exception("Unrecognized file type: " + file)
