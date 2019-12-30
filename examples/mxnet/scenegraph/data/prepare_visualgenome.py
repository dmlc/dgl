"""Prepare Visual Genome datasets"""
import os
import shutil
import argparse
import zipfile
import random
import json
import tqdm
import pickle
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
        ('https://visualgenome.org/static/data/dataset/attribute_synsets.json.zip',
         '54d2b2e33a3bc2a81fbd135c143c973a78f4e29a'),
        ('https://visualgenome.org/static/data/dataset/objects.json.zip',
         'ef3173d5e9ba4be7ad10b32c7619ab7ad51c125e'),
        ('https://visualgenome.org/static/data/dataset/object_synsets.json.zip',
         '8729a12c455d7230362ec599352d4529015ba6e2'),
        ('https://visualgenome.org/static/data/dataset/relationships.json.zip',
         'a1c6873f98e6ef4cbf9c26a810444bfb4cdb69c6'),
        ('https://visualgenome.org/static/data/dataset/relationship_synsets.json.zip',
         '18328be12b7e69a8bd9092c9af885ed2d3511790'),
        ('https://visualgenome.org/static/data/dataset/object_alias.txt',
         '2e61a3f58c391bab4651d6aa1e085af4d19120d0'),
        ('https://visualgenome.org/static/data/dataset/relationship_alias.txt',
         '9042c9f1581656ca8cbf1d347bf4b896a2d85c66'),
    ]
    makedirs(path)
    for url, checksum in _DOWNLOAD_URLS:
        filename = download(url, path=path, overwrite=overwrite, sha1_hash=checksum)
        # extract
        if filename.endswith('zip'):
            with zipfile.ZipFile(filename) as zf:
                zf.extractall(path=path)

def split(path, train_ratio=0.7):
    assert train_ratio > 0 and train_ratio < 1
    vg_100k_path = os.path.join(path, 'VG_100K')
    vg_100k_2_path = os.path.join(path, 'VG_100K_2')
    files_2 = os.listdir(vg_100k_2_path)
    for fl in files_2:
        shutil.move(os.path.join(vg_100k_2_path, fl),
                    os.path.join(vg_100k_path, fl))

    train_path = os.path.join(path, 'train')
    val_path = os.path.join(path, 'val')
    if os.path.isdir(train_path) and os.path.isdir(val_path):
        return
    if not os.path.isdir(train_path):
        makedirs(train_path)
    if not os.path.isdir(val_path):
        makedirs(val_path)
    files = sorted(os.listdir(vg_100k_path))
    random.shuffle(files)
    N = len(files)
    train_n = int(N * train_ratio)
    val_n = N - train_n
    for i, fl in tqdm.tqdm(enumerate(files)):
        if i < train_n:
            shutil.copy(os.path.join(vg_100k_path, fl),
                        os.path.join(train_path, fl))
        else:
            shutil.copy(os.path.join(vg_100k_path, fl),
                        os.path.join(val_path, fl))

def split_json(path, src_json, train_json, val_json):
    json_path = os.path.join(path, src_json)
    train_path = os.path.join(path, 'train')
    val_path = os.path.join(path, 'val')
    train_json_path = os.path.join(path, train_json)
    val_json_path = os.path.join(path, val_json)

    if os.path.exists(train_json_path) and os.path.exists(val_json_path):
        return

    train_img_id = os.listdir(train_path)
    train_img_id = [int(i.split('.')[0]) for i in train_img_id]
    val_img_id = os.listdir(val_path)
    val_img_id = [int(i.split('.')[0]) for i in val_img_id]
    all_img_id = train_img_id + val_img_id
    max_img_id = max(all_img_id)
    img_id_map = [-1 for i in range(max_img_id+1)]
    for i in train_img_id:
        img_id_map[i] = 0
    for i in val_img_id:
        img_id_map[i] = 1

    with open(json_path, 'r') as f:
        tmp = f.read()
        obj_json = json.loads(tmp)

    N = len(obj_json)
    obj_json_train = []
    obj_json_val = []
    for obj in tqdm.tqdm(obj_json):
        status = img_id_map[obj['image_id']]
        if status == 0:
            obj_json_train.append(obj)
        elif status == 1:
            obj_json_val.append(obj)

    with open(train_json_path, 'w') as f:
        json.dump(obj_json_train, f)
    with open(val_json_path, 'w') as f:
        json.dump(obj_json_val, f)

def extract_obj_rel_classes(path):
    obj_path = os.path.join(path, 'objects.json')
    rel_path = os.path.join(path, 'relationships.json')

    with open(obj_path) as f:
        tmp = f.read()
        obj_dict = json.loads(tmp)
    with open(rel_path) as f:
        tmp = f.read()
        rel_dict = json.loads(tmp)

    obj_ctr = {}
    for it in obj_dict:
        for r in it['objects']:
            if len(r['synsets']) > 0:
                k = r['synsets'][0].split('.')[0]
                if k in obj_ctr:
                    obj_ctr[k] += 1
                else:
                    obj_ctr[k] = 1

    rel_ctr = {}
    for it in rel_dict:
        for r in it['relationships']:
            if len(r['synsets']) > 0:
                k = r['synsets'][0].split('.')[0]
                if k in rel_ctr:
                    rel_ctr[k] += 1
                else:
                    rel_ctr[k] = 1
    vg_obj_classes = sorted(obj_ctr, key=obj_ctr.get, reverse=True)
    vg_rel_classes = sorted(rel_ctr, key=rel_ctr.get, reverse=True)
    return vg_obj_classes, vg_rel_classes

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

    split(path)
    split_json(path, 'objects.json', 'objects_train.json', 'objects_val.json')
    split_json(path, 'relationships.json', 'relationships_train.json', 'relationships_val.json')

    vg_obj_classes, vg_rel_classes = extract_obj_rel_classes(path)
    classes = (vg_obj_classes, vg_rel_classes)
    with open(os.path.join(path, 'classes.pkl'), 'wb') as f:
        pickle.dump(classes, f)

    # make symlink
    makedirs(os.path.expanduser('~/.mxnet/datasets'))
    if os.path.isdir(_TARGET_DIR):
        os.remove(_TARGET_DIR)
    os.symlink(path, _TARGET_DIR)
