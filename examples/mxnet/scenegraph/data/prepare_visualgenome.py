"""Prepare Visual Genome datasets"""
import argparse
import json
import os
import pickle
import random
import shutil
import zipfile

import tqdm
from gluoncv.utils import download, makedirs

_TARGET_DIR = os.path.expanduser("~/.mxnet/datasets/visualgenome")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Initialize Visual Genome dataset.",
        epilog="Example: python visualgenome.py --download-dir ~/visualgenome",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default="~/visualgenome/",
        help="dataset directory on disk",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="disable automatic download if set",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite downloaded files if set, in case they are corrupted",
    )
    args = parser.parse_args()
    return args


def download_vg(path, overwrite=False):
    _DOWNLOAD_URLS = [
        (
            "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
            "a055367f675dd5476220e9b93e4ca9957b024b94",
        ),
        (
            "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip",
            "2add3aab77623549e92b7f15cda0308f50b64ecf",
        ),
    ]
    makedirs(path)
    for url, checksum in _DOWNLOAD_URLS:
        filename = download(
            url, path=path, overwrite=overwrite, sha1_hash=checksum
        )
        # extract
        if filename.endswith("zip"):
            with zipfile.ZipFile(filename) as zf:
                zf.extractall(path=path)
    # move all images into folder `VG_100K`
    vg_100k_path = os.path.join(path, "VG_100K")
    vg_100k_2_path = os.path.join(path, "VG_100K_2")
    files_2 = os.listdir(vg_100k_2_path)
    for fl in files_2:
        shutil.move(
            os.path.join(vg_100k_2_path, fl), os.path.join(vg_100k_path, fl)
        )


def download_json(path, overwrite=False):
    url = "https://data.dgl.ai/dataset/vg.zip"
    output = "vg.zip"
    download(url, path=path)
    with zipfile.ZipFile(output) as zf:
        zf.extractall(path=path)
    json_path = os.path.join(path, "vg")
    json_files = os.listdir(json_path)
    for fl in json_files:
        shutil.move(os.path.join(json_path, fl), os.path.join(path, fl))
    os.rmdir(json_path)


if __name__ == "__main__":
    args = parse_args()
    path = os.path.expanduser(args.download_dir)
    if not os.path.isdir(path):
        if args.no_download:
            raise ValueError(
                (
                    "{} is not a valid directory, make sure it is present."
                    ' Or you should not disable "--no-download" to grab it'.format(
                        path
                    )
                )
            )
        else:
            download_vg(path, overwrite=args.overwrite)
            download_json(path, overwrite=args.overwrite)

    # make symlink
    makedirs(os.path.expanduser("~/.mxnet/datasets"))
    if os.path.isdir(_TARGET_DIR):
        os.rmdir(_TARGET_DIR)
    os.symlink(path, _TARGET_DIR)
