import os
import torch as th
import torch.nn as nn
import tqdm


class PBar(object):
    def __enter__(self):
        self.t = None
        return self

    def __call__(self, blockno, readsize, totalsize):
        if self.t is None:
            self.t = tqdm.tqdm(total=totalsize)
        self.t.update(readsize)

    def __exit__(self, exc_type, exc_value, traceback):
        self.t.close()


class AminerDataset(object):
    """
    Download Aminer Dataset from Amazon S3 bucket. 
    """
    def __init__(self, path):

        self.url = 'https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/aminer.zip'

        if not os.path.exists(os.path.join(path, 'aminer')):
            print('File not found. Downloading from', self.url)
            self._download_and_extract(path, 'aminer.zip')

    def _download_and_extract(self, path, filename):
        import shutil, zipfile, zlib
        from tqdm import tqdm
        import urllib.request

        fn = os.path.join(path, filename)

        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path)
        with PBar() as pb:
            urllib.request.urlretrieve(self.url, fn, pb)
        print('Download finished. Unzipping the file...')

        with zipfile.ZipFile(fn) as zf:
            zf.extractall(path)
        print('Unzip finished.')
        self.fn = fn
