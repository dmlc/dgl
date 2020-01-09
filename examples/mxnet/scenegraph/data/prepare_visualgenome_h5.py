"""Prepare Visual Genome datasets"""
import os, shutil, argparse, zipfile, random, json, tqdm, pickle, h5py, cv2, string, floor, pprint
import numpy as np
import h5py as h5

from collections import Counter
from queue import Queue
from threading import Thread, Lock
from math import floor
from gluoncv.utils import download, makedirs

_TARGET_DIR = os.path.expanduser('~/.mxnet/datasets/visualgenome')

def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize Visual Genome dataset.',
        epilog='Example: python visualgenome.py --download-dir ~/visualgenome',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--download-dir', type=str, default='~/visualgenome/',
                        help='dataset directory on disk')
    parser.add_argument('--no-download', action='store_true', help='disable automatic download if set')
    parser.add_argument('--overwrite', action='store_true', help='overwrite downloaded files if set, in case they are corrupted')
    args = parser.parse_args()
    return args

def download_vg(path, overwrite=False):
    _DOWNLOAD_URLS = [
        ('https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip',
         'a055367f675dd5476220e9b93e4ca9957b024b94'),
        ('https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip',
         '2add3aab77623549e92b7f15cda0308f50b64ecf'),
        ('http://svl.stanford.edu/projects/scene-graph/VG/image_data.json',
         'c550d8bf07fe9cbb951c94f1fabd827cc934cefe'),
        ('http://svl.stanford.edu/projects/scene-graph/VG/VG-scene-graph.zip',
         '0bc3904a73388c81346dacec9f0534fc5712f098'),
    ]
    makedirs(path)
    for url, checksum in _DOWNLOAD_URLS:
        filename = download(url, path=path, overwrite=overwrite, sha1_hash=checksum)
        # extract
        if filename.endswith('zip'):
            with zipfile.ZipFile(filename) as zf:
                zf.extractall(path=path)

    img_path = os.path.join(path, 'images') 
    makedirs(os.path.join(path, 'images'))

    path_1 = os.path.join(path, 'VG_100K')
    files_1 = os.listdir(path_1)
    path_2 = os.path.join(path, 'VG_100K_2')
    files_2 = os.listdir(path_2)
    for fl in files_1:
        shutil.move(os.path.join(path_1, fl),
                    os.path.join(img_path, fl))
    for fl in files_2:
        shutil.move(os.path.join(path_2, fl),
                    os.path.join(img_path, fl))

    shutil.move(os.path.join(path, 'data_release', 'objects.json'), path)
    shutil.move(os.path.join(path, 'data_release', 'relationships.json'), path)

if __name__ == '__main__':
    random.seed(2048)
    args = parse_args()
    path = os.path.expanduser(args.download_dir)
    if not os.path.isdir(path):
        if args.no_download:
            raise ValueError(('{} is not a valid directory, make sure it is present.'
                              ' Or you should not disable "--no-download" to grab it'.format(path)))
        else:
            download_vg(path, overwrite=args.overwrite)

    # make symlink
    makedirs(os.path.expanduser('~/.mxnet/datasets'))
    if os.path.isdir(_TARGET_DIR):
        os.remove(_TARGET_DIR)
    os.symlink(path, _TARGET_DIR)
