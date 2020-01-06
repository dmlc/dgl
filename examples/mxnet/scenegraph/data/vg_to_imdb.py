# coding=utf8

import argparse, os, json, string
from queue import Queue
from threading import Thread, Lock

import h5py
import numpy as np
# from scipy.misc import imread, imresize
import cv2

def build_filename_dict(data):
    # First make sure all basenames are unique
    basenames_list = [os.path.basename(img['image_path']) for img in data]
    assert len(basenames_list) == len(set(basenames_list))

    next_idx = 1
    filename_to_idx, idx_to_filename = {}, {}
    for img in data:
        filename = os.path.basename(img['image_path'])
        filename_to_idx[filename] = next_idx
        idx_to_filename[next_idx] = filename
        next_idx += 1
    return filename_to_idx, idx_to_filename


def encode_filenames(data, filename_to_idx):
    filename_idxs = []
    for img in data:
        filename = os.path.basename(img['image_path'])
        idx = filename_to_idx[filename]
        filename_idxs.append(idx)
    return np.asarray(filename_idxs, dtype=np.int32)


def add_images(im_data, h5_file, args):
    fns = []; ids = []; idx = []
    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    for i, img in enumerate(im_data):
        basename =  str(img['image_id']) + '.jpg'
        if basename in corrupted_ims:
            continue
            
        filename = os.path.join(args.image_dir, basename)
        if os.path.exists(filename):
            fns.append(filename)
            ids.append(img['image_id'])
            idx.append(i)

    ids = np.array(ids, dtype=np.int32)
    idx = np.array(idx, dtype=np.int32)
    h5_file.create_dataset('image_ids', data=ids)
    h5_file.create_dataset('valid_idx', data=idx)

    num_images = len(fns)

    shape = (num_images, 3, args.image_size, args.image_size)
    image_dset = h5_file.create_dataset('images', shape, dtype=np.uint8)
    original_heights = np.zeros(num_images, dtype=np.int32)
    original_widths = np.zeros(num_images, dtype=np.int32)
    image_heights = np.zeros(num_images, dtype=np.int32)
    image_widths = np.zeros(num_images, dtype=np.int32)

    lock = Lock()
    q = Queue()
    for i, fn in enumerate(fns):
        q.put((i, fn))

    def worker():
        while True:
            i, filename = q.get()

            if i % 10000 == 0:
                print('processing %i images...' % i)
            img = cv2.imread(filename)
            # handle grayscale
            if img.ndim == 2:
                img = img[:, :, None][:, :, [0, 0, 0]]
            H0, W0 = img.shape[0], img.shape[1]
            ratio = float(args.image_size) / max(H0, W0)
            nH = int(H0 * ratio)
            nW = int(W0 * ratio)
            img = cv2.resize(img, (nW, nH), interpolation = cv2.INTER_LINEAR)
            H, W = img.shape[0], img.shape[1]
            # swap rgb to bgr. This can't be the best way right? #fail
            '''
            r = img[:,:,0].copy()
            img[:,:,0] = img[:,:,2]
            img[:,:,2] = r
            '''

            lock.acquire()
            original_heights[i] = H0
            original_widths[i] = W0
            image_heights[i] = H
            image_widths[i] = W
            image_dset[i, :, :H, :W] = img.transpose(2, 0, 1)
            lock.release()
            q.task_done()

    for i in range(args.num_workers):
        t = Thread(target=worker)
        t.daemon = True
        t.start()

    q.join()

    h5_file.create_dataset('image_heights', data=image_heights)
    h5_file.create_dataset('image_widths', data=image_widths)
    h5_file.create_dataset('original_heights', data=original_heights)
    h5_file.create_dataset('original_widths', data=original_widths)

    return fns


def main(args):
    im_metadata = json.load(open(args.metadata_input))
    h5_fn = 'imdb_' + str(args.image_size) + '.h5'
    # write the h5 file
    h5_file = os.path.join(args.imh5_dir, h5_fn)
    f = h5py.File(h5_file, 'w')
    # load images
    im_fns = add_images(im_metadata, f, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', default='VG/images')
    parser.add_argument('--image_size', default=1024, type=int)
    parser.add_argument('--imh5_dir', default='.')
    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--metadata_input', default='VG/image_data.json', type=str)

    args = parser.parse_args()
    main(args)
