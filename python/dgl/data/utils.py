"""Dataset utilities."""
from __future__ import absolute_import

import os, sys
import hashlib
import warnings
import zipfile
import tarfile
try:
    import requests
except ImportError:
    class requests_failed_to_import(object):
        pass
    requests = requests_failed_to_import

__all__ = ['download', 'check_sha1', 'extract_archive', 'get_download_dir']

def _get_dgl_url(file_url):
    """Get DGL online url for download."""
    dgl_repo_url = 'https://s3.us-east-2.amazonaws.com/dgl.ai/'
    repo_url = os.environ.get('DGL_REPO', dgl_repo_url)
    if repo_url[-1] != '/':
        repo_url = repo_url + '/'
    return repo_url + file_url


def download(url, path=None, overwrite=False, sha1_hash=None, retries=5, verify_ssl=True):
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
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    retries : integer, default 5
        The number of times to attempt downloading in case of failure or non 200 return codes.
    verify_ssl : bool, default True
        Verify SSL certificates.

    Returns
    -------
    str
        The file path of the downloaded file.
    """
    if path is None:
        fname = url.split('/')[-1]
        # Empty filenames are invalid
        assert fname, 'Can\'t construct file-name from this URL. ' \
            'Please set the `path` option manually.'
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path
    assert retries >= 0, "Number of retries should be at least 0"

    if not verify_ssl:
        warnings.warn(
            'Unverified HTTPS request is being made (verify_ssl=False). '
            'Adding certificate verification is strongly advised.')

    if overwrite or not os.path.exists(fname) or (sha1_hash and not check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        while retries+1 > 0:
            # Disable pyling too broad Exception
            # pylint: disable=W0703
            try:
                print('Downloading %s from %s...'%(fname, url))
                r = requests.get(url, stream=True, verify=verify_ssl)
                if r.status_code != 200:
                    raise RuntimeError("Failed downloading url %s"%url)
                with open(fname, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk: # filter out keep-alive new chunks
                            f.write(chunk)
                if sha1_hash and not check_sha1(fname, sha1_hash):
                    raise UserWarning('File {} is downloaded but the content hash does not match.'\
                                      ' The repo may be outdated or download may be incomplete. '\
                                      'If the "repo_url" is overridden, consider switching to '\
                                      'the default repo.'.format(fname))
                break
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    raise e
                else:
                    print("download failed, retrying, {} attempt{} left"
                          .format(retries, 's' if retries > 1 else ''))

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
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash

def extract_archive(file, target_dir):
    """Extract archive file.

    Parameters
    ----------
    file : str
        Absolute path of the archive file.
    target_dir : str
        Target directory of the archive to be uncompressed.
    """
    if os.path.exists(target_dir):
        return
    if file.endswith('.gz') or file.endswith('.tar') or file.endswith('.tgz'):
        archive = tarfile.open(file, 'r')
    elif file.endswith('.zip'):
        archive = zipfile.ZipFile(file, 'r')
    else:
        raise Exception('Unrecognized file type: ' + file)
    print('Extracting file to {}'.format(target_dir))
    archive.extractall(path=target_dir)
    archive.close()

def get_download_dir():
    """Get the absolute path to the download directory.

    Returns
    -------
    dirname : str
        Path to the download directory
    """
    default_dir = os.path.join(os.path.expanduser('~'), '.dgl')
    dirname = os.environ.get('DGL_DOWNLOAD_DIR', default_dir)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname
