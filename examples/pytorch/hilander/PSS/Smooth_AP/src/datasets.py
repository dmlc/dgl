# repo originally forked from https://github.com/Confusezius/Deep-Metric-Learning-Baselines

################# LIBRARIES ###############################
import pickle
import warnings

from numpy.core.arrayprint import IntegerFormat

warnings.filterwarnings("ignore")

import copy
import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

"""============================================================================"""


################ FUNCTION TO RETURN ALL DATALOADERS NECESSARY ####################
def give_dataloaders(dataset, trainset, testset, opt, cluster_path=""):
    """
    Args:
        dataset: string, name of dataset for which the dataloaders should be returned.
        opt:     argparse.Namespace, contains all training-specific parameters.
    Returns:
        dataloaders: dict of dataloaders for training, testing and evaluation on training.
    """
    # Dataset selection
    if opt.dataset == "Inaturalist":
        if opt.finetune:
            datasets = give_inat_datasets_finetune_1head(
                testset, cluster_path, opt
            )
        else:
            if opt.get_features:
                datasets = give_inaturalist_datasets_for_features(opt)
            else:
                datasets = give_inaturalist_datasets(opt)
    else:
        raise Exception("No Dataset >{}< available!".format(dataset))

    # Move datasets to dataloaders.
    dataloaders = {}

    for key, dataset in datasets.items():
        if (
            isinstance(dataset, TrainDatasetsmoothap)
            or isinstance(dataset, TrainDatasetsmoothap1Head)
        ) and key in ["training", "clustering"]:
            dataloaders[key] = torch.utils.data.DataLoader(
                dataset,
                batch_size=opt.bs,
                num_workers=opt.kernels,
                sampler=torch.utils.data.SequentialSampler(dataset),
                pin_memory=True,
                drop_last=True,
            )
        else:
            is_val = dataset.is_validation
            if key == "training" or key == "clustering":
                dataloaders[key] = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=opt.bs,
                    num_workers=opt.kernels,
                    shuffle=not is_val,
                    pin_memory=True,
                    drop_last=not is_val,
                )
            else:
                dataloaders[key] = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=opt.bs,
                    num_workers=6,
                    shuffle=not is_val,
                    pin_memory=True,
                    drop_last=not is_val,
                )
    return dataloaders


def give_inaturalist_datasets(opt):
    """
    This function generates a training, testing and evaluation dataloader for Metric Learning on the Inaturalist 2018 dataset.
    For Metric Learning, training and test sets are provided by given json files. Will define a train and test split
    So no random shuffling of classes.

    Args:
        opt: argparse.Namespace, contains all traininig-specific parameters.
    Returns:
        dict of PyTorch datasets for training, testing and evaluation.
    """
    # Load text-files containing classes and imagepaths.
    # Generate image_dicts of shape {class_idx:[list of paths to images belong to this class] ...}
    train_image_dict, val_image_dict, test_image_dict = {}, {}, {}
    with open(os.path.join(opt.source_path, opt.trainset)) as f:
        FileLines = f.readlines()
        FileLines = [x.strip() for x in FileLines]

        for entry in FileLines:
            info = entry.split("/")
            if "/".join([info[-3], info[-2]]) not in train_image_dict:
                train_image_dict["/".join([info[-3], info[-2]])] = []
            train_image_dict["/".join([info[-3], info[-2]])].append(
                os.path.join(opt.source_path, entry)
            )

    with open(os.path.join(opt.source_path, opt.testset)) as f:
        FileLines = f.readlines()
        FileLines = [x.strip() for x in FileLines]

        for entry in FileLines:
            info = entry.split("/")
            if "/".join([info[-3], info[-2]]) not in val_image_dict:
                val_image_dict["/".join([info[-3], info[-2]])] = []
            val_image_dict["/".join([info[-3], info[-2]])].append(
                os.path.join(opt.source_path, entry)
            )

    with open(os.path.join(opt.source_path, opt.testset)) as f:
        FileLines = f.readlines()
        FileLines = [x.strip() for x in FileLines]

        for entry in FileLines:
            info = entry.split("/")
            if "/".join([info[-3], info[-2]]) not in test_image_dict:
                test_image_dict["/".join([info[-3], info[-2]])] = []
            test_image_dict["/".join([info[-3], info[-2]])].append(
                os.path.join(opt.source_path, entry)
            )

    new_train_dict = {}
    class_ind_ind = 0
    for cate in train_image_dict:
        new_train_dict["te/%d" % class_ind_ind] = train_image_dict[cate]
        class_ind_ind += 1
    train_image_dict = new_train_dict

    train_dataset = TrainDatasetsmoothap(train_image_dict, opt)

    val_dataset = BaseTripletDataset(val_image_dict, opt, is_validation=True)
    eval_dataset = BaseTripletDataset(test_image_dict, opt, is_validation=True)

    # train_dataset.conversion       = conversion
    # val_dataset.conversion         = conversion
    # eval_dataset.conversion        = conversion

    return {
        "training": train_dataset,
        "testing": val_dataset,
        "evaluation": eval_dataset,
    }


def give_inaturalist_datasets_for_features(opt):
    """
    This function generates a training, testing and evaluation dataloader for Metric Learning on the Inaturalist 2018 dataset.
    For Metric Learning, training and test sets are provided by given json files. Will define a train and test split
    So no random shuffling of classes.

    Args:
        opt: argparse.Namespace, contains all traininig-specific parameters.
    Returns:
        dict of PyTorch datasets for training, testing and evaluation.
    """
    # Load text-files containing classes and imagepaths.
    # Generate image_dicts of shape {class_idx:[list of paths to images belong to this class] ...}
    train_image_dict, test_image_dict, eval_image_dict = {}, {}, {}

    if opt.iter > 0:
        with open(os.path.join(opt.cluster_path), "rb") as clusterf:
            (
                path2idx,
                global_features,
                global_pred_labels,
                gt_labels,
                masks,
            ) = pickle.load(clusterf)
            gt_labels = gt_labels + len(np.unique(global_pred_labels))

            for path, idx in path2idx.items():
                if global_pred_labels[idx] == -1:
                    if "te/%d" % gt_labels[idx] not in test_image_dict:
                        test_image_dict["te/%d" % gt_labels[idx]] = []
                    test_image_dict["te/%d" % gt_labels[idx]].append(path)
                else:
                    if (
                        "te/%d" % global_pred_labels[idx]
                        not in train_image_dict
                    ):
                        train_image_dict["te/%d" % global_pred_labels[idx]] = []
                    train_image_dict["te/%d" % global_pred_labels[idx]].append(
                        path
                    )
                    if "te/%d" % global_pred_labels[idx] not in test_image_dict:
                        test_image_dict["te/%d" % global_pred_labels[idx]] = []
                    test_image_dict["te/%d" % global_pred_labels[idx]].append(
                        path
                    )
    else:
        with open(os.path.join(opt.source_path, opt.trainset)) as f:
            FileLines = f.readlines()
            FileLines = [x.strip() for x in FileLines]

            for entry in FileLines:
                info = entry.split("/")
                if "/".join([info[-3], info[-2]]) not in train_image_dict:
                    train_image_dict["/".join([info[-3], info[-2]])] = []
                train_image_dict["/".join([info[-3], info[-2]])].append(
                    os.path.join(opt.source_path, entry)
                )

        with open(os.path.join(opt.source_path, opt.all_trainset)) as f:
            FileLines = f.readlines()
            FileLines = [x.strip() for x in FileLines]
            for entry in FileLines:
                info = entry.split("/")
                if "/".join([info[-3], info[-2]]) not in test_image_dict:
                    test_image_dict["/".join([info[-3], info[-2]])] = []
                test_image_dict["/".join([info[-3], info[-2]])].append(
                    os.path.join(opt.source_path, entry)
                )

    with open(os.path.join(opt.source_path, opt.testset)) as f:
        FileLines = f.readlines()
        FileLines = [x.strip() for x in FileLines]

        for entry in FileLines:
            info = entry.split("/")
            if "/".join([info[-3], info[-2]]) not in eval_image_dict:
                eval_image_dict["/".join([info[-3], info[-2]])] = []
            eval_image_dict["/".join([info[-3], info[-2]])].append(
                os.path.join(opt.source_path, entry)
            )

    new_train_dict = {}
    class_ind_ind = 0
    for cate in train_image_dict:
        new_train_dict["te/%d" % class_ind_ind] = train_image_dict[cate]
        class_ind_ind += 1
    train_image_dict = new_train_dict

    new_test_dict = {}
    class_ind_ind = 0
    for cate in test_image_dict:
        new_test_dict["te/%d" % class_ind_ind] = test_image_dict[cate]
        class_ind_ind += 1
    test_image_dict = new_test_dict

    new_eval_dict = {}
    class_ind_ind = 0
    for cate in eval_image_dict:
        new_eval_dict["te/%d" % class_ind_ind] = eval_image_dict[cate]
        class_ind_ind += 1
    eval_image_dict = new_eval_dict

    train_dataset = BaseTripletDataset(
        train_image_dict, opt, is_validation=True
    )
    test_dataset = BaseTripletDataset(test_image_dict, opt, is_validation=True)
    eval_dataset = BaseTripletDataset(eval_image_dict, opt, is_validation=True)

    # train_dataset.conversion       = conversion
    # val_dataset.conversion         = conversion
    # eval_dataset.conversion        = conversion

    return {
        "training": train_dataset,
        "testing": test_dataset,
        "eval": eval_dataset,
    }


def give_inat_datasets_finetune_1head(testset, cluster_label_path, opt):
    """
    This function generates a training, testing and evaluation dataloader for Metric Learning on the Inaturalist 2018 dataset.
    For Metric Learning, training and test sets are provided by given json files. Will define a train and test split
    So no random shuffling of classes.

    Args:
        opt: argparse.Namespace, contains all traininig-specific parameters.
    Returns:
        dict of PyTorch datasets for training, testing and evaluation.
    """
    # Load cluster labels from hilander results.
    import pickle

    train_image_dict, val_image_dict, cluster_image_dict = {}, {}, {}
    with open(cluster_label_path, "rb") as clusterf:
        (
            path2idx,
            global_features,
            global_pred_labels,
            gt_labels,
            masks,
        ) = pickle.load(clusterf)

        for path, idx in path2idx.items():
            if global_pred_labels[idx] == -1:
                continue
            else:
                if "te/%d" % global_pred_labels[idx] not in train_image_dict:
                    train_image_dict["te/%d" % global_pred_labels[idx]] = []
                train_image_dict["te/%d" % global_pred_labels[idx]].append(path)

    with open(os.path.join(opt.source_path, testset)) as f:
        FileLines = f.readlines()
        FileLines = [x.strip() for x in FileLines]

        for entry in FileLines:
            info = entry.split("/")
            if "/".join([info[-3], info[-2]]) not in val_image_dict:
                val_image_dict["/".join([info[-3], info[-2]])] = []
            val_image_dict["/".join([info[-3], info[-2]])].append(
                os.path.join(opt.source_path, entry)
            )

    train_dataset = TrainDatasetsmoothap(train_image_dict, opt)

    val_dataset = BaseTripletDataset(val_image_dict, opt, is_validation=True)

    # train_dataset.conversion       = conversion
    # val_dataset.conversion         = conversion
    # eval_dataset.conversion        = conversion

    return {
        "training": train_dataset,
        "testing": val_dataset,
        "evaluation": val_dataset,
    }


################## BASIC PYTORCH DATASET USED FOR ALL DATASETS ##################################
class BaseTripletDataset(Dataset):
    """
    Dataset class to provide (augmented) correctly prepared training samples corresponding to standard DML literature.
    This includes normalizing to ImageNet-standards, and Random & Resized cropping of shapes 224 for ResNet50 and 227 for
    GoogLeNet during Training. During validation, only resizing to 256 or center cropping to 224/227 is performed.
    """

    def __init__(
        self, image_dict, opt, samples_per_class=8, is_validation=False
    ):
        """
        Dataset Init-Function.

        Args:
            image_dict:         dict, Dictionary of shape {class_idx:[list of paths to images belong to this class] ...} providing all the training paths and classes.
            opt:                argparse.Namespace, contains all training-specific parameters.
            samples_per_class:  Number of samples to draw from one class before moving to the next when filling the batch.
            is_validation:      If is true, dataset properties for validation/testing are used instead of ones for training.
        Returns:
            Nothing!
        """
        # Define length of dataset
        self.n_files = np.sum(
            [len(image_dict[key]) for key in image_dict.keys()]
        )

        self.is_validation = is_validation

        self.pars = opt
        self.image_dict = image_dict

        self.avail_classes = sorted(list(self.image_dict.keys()))

        # Convert image dictionary from classname:content to class_idx:content, because the initial indices are not necessarily from 0 - <n_classes>.
        self.image_dict = {
            i: self.image_dict[key] for i, key in enumerate(self.avail_classes)
        }
        self.avail_classes = sorted(list(self.image_dict.keys()))

        # Init. properties that are used when filling up batches.
        if not self.is_validation:
            self.samples_per_class = samples_per_class
            # Select current class to sample images from up to <samples_per_class>
            self.current_class = np.random.randint(len(self.avail_classes))
            self.classes_visited = [self.current_class, self.current_class]
            self.n_samples_drawn = 0

        # Data augmentation/processing methods.
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        transf_list = []
        if not self.is_validation:
            transf_list.extend(
                [
                    transforms.RandomResizedCrop(size=224)
                    if opt.arch == "resnet50"
                    else transforms.RandomResizedCrop(size=227),
                    transforms.RandomHorizontalFlip(0.5),
                ]
            )
        else:
            transf_list.extend(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224)
                    if opt.arch == "resnet50"
                    else transforms.CenterCrop(227),
                ]
            )

        transf_list.extend([transforms.ToTensor(), normalize])
        self.transform = transforms.Compose(transf_list)

        # Convert Image-Dict to list of (image_path, image_class). Allows for easier direct sampling.
        self.image_list = [
            [(x, key) for x in self.image_dict[key]]
            for key in self.image_dict.keys()
        ]
        self.image_list = [x for y in self.image_list for x in y]

        # Flag that denotes if dataset is called for the first time.
        self.is_init = True

    def ensure_3dim(self, img):
        """
        Function that ensures that the input img is three-dimensional.

        Args:
            img: PIL.Image, image which is to be checked for three-dimensionality (i.e. if some images are black-and-white in an otherwise coloured dataset).
        Returns:
            Checked PIL.Image img.
        """
        if len(img.size) == 2:
            img = img.convert("RGB")
        return img

    def __getitem__(self, idx):
        """
        Args:
            idx: Sample idx for training sample
        Returns:
            tuple of form (sample_class, torch.Tensor() of input image)
        """
        if self.pars.loss == "smoothap" or self.pars.loss == "smoothap_element":
            if self.is_init:
                # self.current_class = self.avail_classes[idx%len(self.avail_classes)]
                self.is_init = False

            if not self.is_validation:
                if self.samples_per_class == 1:
                    return self.image_list[idx][-1], self.transform(
                        self.ensure_3dim(Image.open(self.image_list[idx][0]))
                    )

                if self.n_samples_drawn == self.samples_per_class:
                    # Once enough samples per class have been drawn, we choose another class to draw samples from.
                    # Note that we ensure with self.classes_visited that no class is chosen if it had been chosen
                    # previously or one before that.
                    counter = copy.deepcopy(self.avail_classes)
                    for prev_class in self.classes_visited:
                        if prev_class in counter:
                            counter.remove(prev_class)

                    self.current_class = counter[idx % len(counter)]
                    # self.classes_visited = self.classes_visited[1:]+[self.current_class]
                    # EDIT -> there can be no class repeats
                    self.classes_visited = self.classes_visited + [
                        self.current_class
                    ]
                    self.n_samples_drawn = 0

                class_sample_idx = idx % len(
                    self.image_dict[self.current_class]
                )
                self.n_samples_drawn += 1

                out_img = self.transform(
                    self.ensure_3dim(
                        Image.open(
                            self.image_dict[self.current_class][
                                class_sample_idx
                            ]
                        )
                    )
                )
                return self.current_class, out_img
            else:
                return self.image_list[idx][-1], self.transform(
                    self.ensure_3dim(Image.open(self.image_list[idx][0]))
                )
        else:
            if self.is_init:
                self.current_class = self.avail_classes[
                    idx % len(self.avail_classes)
                ]
                self.is_init = False
            if not self.is_validation:
                if self.samples_per_class == 1:
                    return self.image_list[idx][-1], self.transform(
                        self.ensure_3dim(Image.open(self.image_list[idx][0]))
                    )

                if self.n_samples_drawn == self.samples_per_class:
                    # Once enough samples per class have been drawn, we choose another class to draw samples from.
                    # Note that we ensure with self.classes_visited that no class is chosen if it had been chosen
                    # previously or one before that.
                    counter = copy.deepcopy(self.avail_classes)
                    for prev_class in self.classes_visited:
                        if prev_class in counter:
                            counter.remove(prev_class)

                    self.current_class = counter[idx % len(counter)]
                    self.classes_visited = self.classes_visited[1:] + [
                        self.current_class
                    ]
                    self.n_samples_drawn = 0

                class_sample_idx = idx % len(
                    self.image_dict[self.current_class]
                )
                self.n_samples_drawn += 1

                out_img = self.transform(
                    self.ensure_3dim(
                        Image.open(
                            self.image_dict[self.current_class][
                                class_sample_idx
                            ]
                        )
                    )
                )
                return self.current_class, out_img
            else:
                return self.image_list[idx][-1], self.transform(
                    self.ensure_3dim(Image.open(self.image_list[idx][0]))
                )

    def __len__(self):
        return self.n_files


flatten = lambda l: [item for sublist in l for item in sublist]

######################## dataset for SmoothAP regular training ##################################


class TrainDatasetsmoothap(Dataset):
    """
    This dataset class allows mini-batch formation pre-epoch, for greater speed

    """

    def __init__(self, image_dict, opt):
        """
        Args:
            image_dict: two-level dict, `super_dict[super_class_id][class_id]` gives the list of
                        image paths having the same super-label and class label
        """
        self.image_dict = image_dict
        self.dataset_name = opt.dataset
        self.batch_size = opt.bs
        self.samples_per_class = opt.samples_per_class
        for sub in self.image_dict:
            newsub = []
            for instance in self.image_dict[sub]:
                newsub.append((sub, instance))
            self.image_dict[sub] = newsub

        # checks
        # provide avail_classes
        self.avail_classes = [*self.image_dict]
        # Data augmentation/processing methods.
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        transf_list = []

        transf_list.extend(
            [
                transforms.RandomResizedCrop(size=224)
                if opt.arch in ["resnet50", "resnet50_mcn"]
                else transforms.RandomResizedCrop(size=227),
                transforms.RandomHorizontalFlip(0.5),
            ]
        )
        transf_list.extend([transforms.ToTensor(), normalize])
        self.transform = transforms.Compose(transf_list)

        self.reshuffle()

    def ensure_3dim(self, img):
        if len(img.size) == 2:
            img = img.convert("RGB")
        return img

    def reshuffle(self):
        image_dict = copy.deepcopy(self.image_dict)
        print("shuffling data")
        for sub in image_dict:
            random.shuffle(image_dict[sub])

        classes = [*image_dict]
        random.shuffle(classes)
        total_batches = []
        batch = []
        finished = 0
        while finished == 0:
            for sub_class in classes:
                if (len(image_dict[sub_class]) >= self.samples_per_class) and (
                    len(batch) < self.batch_size / self.samples_per_class
                ):
                    batch.append(
                        image_dict[sub_class][: self.samples_per_class]
                    )
                    image_dict[sub_class] = image_dict[sub_class][
                        self.samples_per_class :
                    ]

            if len(batch) == self.batch_size / self.samples_per_class:
                total_batches.append(batch)
                batch = []
            else:
                finished = 1

        random.shuffle(total_batches)
        self.dataset = flatten(flatten(total_batches))

    def __getitem__(self, idx):
        # we use SequentialSampler together with SuperLabelTrainDataset,
        # so idx==0 indicates the start of a new epoch
        batch_item = self.dataset[idx]

        if self.dataset_name == "Inaturalist":
            cls = int(batch_item[0].split("/")[1])

        else:
            cls = batch_item[0]
        img = Image.open(batch_item[1])
        return cls, self.transform(self.ensure_3dim(img))

    def __len__(self):
        return len(self.dataset)


class TrainDatasetsmoothap1Head(Dataset):
    """
    This dataset class allows mini-batch formation pre-epoch, for greater speed

    """

    def __init__(self, image_dict_L, image_dict_U, opt):
        """
        Args:
            image_dict: two-level dict, `super_dict[super_class_id][class_id]` gives the list of
                        image paths having the same super-label and class label
        """
        self.image_dict_L = image_dict_L
        self.image_dict_U = image_dict_U
        self.dataset_name = opt.dataset
        self.batch_size = opt.bs
        self.samples_per_class = opt.samples_per_class
        for sub_L in self.image_dict_L:
            newsub_L = []
            for instance in self.image_dict_L[sub_L]:
                newsub_L.append((sub_L, instance))
            self.image_dict_L[sub_L] = newsub_L

        for sub_U in self.image_dict_U:
            newsub_U = []
            for instance in self.image_dict_U[sub_U]:
                newsub_U.append((sub_U, instance))
            self.image_dict_U[sub_U] = newsub_U

        # checks
        # provide avail_classes
        self.avail_classes = [*self.image_dict_L] + [*self.image_dict_U]
        # Data augmentation/processing methods.
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        transf_list = []

        transf_list.extend(
            [
                transforms.RandomResizedCrop(size=224)
                if opt.arch in ["resnet50", "resnet50_mcn"]
                else transforms.RandomResizedCrop(size=227),
                transforms.RandomHorizontalFlip(0.5),
            ]
        )
        transf_list.extend([transforms.ToTensor(), normalize])
        self.transform = transforms.Compose(transf_list)

        self.reshuffle()

    def sample_same_size(self):
        image_dict = copy.deepcopy(self.image_dict_L)

        L_size = 0
        for sub_L in self.image_dict_L:
            L_size += len(self.image_dict_L[sub_L])

        U_size = 0
        classes_U = [*self.image_dict_U]
        # while U_size < len(list(self.image_dict_U)) and U_size < L_size:
        while len(classes_U) != 0:
            sub_U = random.choice(classes_U)
            classes_U.remove(sub_U)
            sub_U_size = len(self.image_dict_U[sub_U])
            if sub_U in [*image_dict]:
                image_dict[sub_U].extend(self.image_dict_U[sub_U])
            else:
                image_dict[sub_U] = self.image_dict_U[sub_U]
            U_size += sub_U_size
        return image_dict

    def ensure_3dim(self, img):
        if len(img.size) == 2:
            img = img.convert("RGB")
        return img

    def reshuffle(self):
        image_dict = self.sample_same_size()
        print("shuffling data")
        for sub in image_dict:
            random.shuffle(image_dict[sub])

        classes = [*image_dict]
        random.shuffle(classes)
        total_batches = []
        batch = []
        finished = 0
        while finished == 0:
            for sub_class in classes:
                if (len(image_dict[sub_class]) >= self.samples_per_class) and (
                    len(batch) < self.batch_size / self.samples_per_class
                ):
                    batch.append(
                        image_dict[sub_class][: self.samples_per_class]
                    )
                    image_dict[sub_class] = image_dict[sub_class][
                        self.samples_per_class :
                    ]

            if len(batch) == self.batch_size / self.samples_per_class:
                total_batches.append(batch)
                batch = []
            else:
                finished = 1

        random.shuffle(total_batches)
        self.dataset = flatten(flatten(total_batches))

    def __getitem__(self, idx):
        # we use SequentialSampler together with SuperLabelTrainDataset,
        # so idx==0 indicates the start of a new epoch
        batch_item = self.dataset[idx]

        if self.dataset_name == "Inaturalist":
            cls = int(batch_item[0].split("/")[1])
        else:
            cls = batch_item[0]
        img = Image.open(str(batch_item[1]))
        return cls, self.transform(self.ensure_3dim(img))

    def __len__(self):
        return len(self.dataset)
