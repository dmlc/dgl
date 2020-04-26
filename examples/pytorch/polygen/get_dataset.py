import os
import sys
import numpy as np


def get_all_obj_file_in_folder(data_dir, all_list_path):
    class_dirs = [os.path.join(data_dir, cdir) for cdir in os.listdir(data_dir) if 
                 os.path.isdir(os.path.join(data_dir, cdir))]
    print (class_dirs)
    with open(all_list_path, 'w') as f:
        for class_dir in class_dirs:
            file_dirs = [os.path.join(class_dir, cdir) for cdir in os.listdir(class_dir) if 
                         os.path.isdir(os.path.join(class_dir, cdir))]
            for f_dir in file_dirs:
                obj_file  = os.path.join(os.path.join(f_dir, 'models'), 'model_normalized.obj')
                if os.path.isfile(obj_file):
                    f.write(obj_file+'\n')

def train_test_split(all_list_path):
    train_part = 0.95

    with open(all_list_path, 'r') as f:
        all_list = [line.strip() for line in f.readlines()]
    num_files = len(all_list)
    num_files_train = int(num_files * train_part)
    arr = np.arange(num_files)
    np.random.shuffle(all_list)
    train_list = all_list[:num_files_train]
    test_list = all_list[num_files_train:]
    with open(all_list_path+'.train', 'w') as f:
        for t_f in train_list:
            f.write(t_f+'\n')
    with open(all_list_path+'.test', 'w') as f:
        for t_f in test_list:
            f.write(t_f+'\n')



if __name__ == '__main__':
    np.random.seed(1111)
    data_dir = '/home/ubuntu/data/new/ShapeNetCore.v2/'
    all_list_path = os.path.join(data_dir, sys.argv[1])
    print (all_list_path)
    #get_all_obj_file_in_folder(data_dir, all_list_path)
    train_test_split(all_list_path)


