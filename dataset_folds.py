import functools

import torch
import torch.utils.data

from frame_field_learning import data_transforms
from lydorn_utils import print_utils


def inria_aerial_train_tile_filter(tile, train_val_split_point):
    return tile["number"] <= train_val_split_point


def inria_aerial_val_tile_filter(tile, train_val_split_point):
    return train_val_split_point < tile["number"]


def get_inria_aerial_folds(config, root_dir, folds):
    from torch_lydorn.torchvision.datasets import InriaAerial

    # --- Online transform done on the host (CPU):
    online_cpu_transform = data_transforms.get_online_cpu_transform(config,
                                                                    augmentations=config["data_aug_params"]["enable"])
    mask_only = config["dataset_params"]["mask_only"]
    kwargs = {
        "pre_process": config["dataset_params"]["pre_process"],
        "transform": online_cpu_transform,
        "patch_size": config["dataset_params"]["data_patch_size"],
        "patch_stride": config["dataset_params"]["input_patch_size"],
        "pre_transform": data_transforms.get_offline_transform_patch(distances=not mask_only, sizes=not mask_only),
        "small": config["dataset_params"]["small"],
        "pool_size": config["num_workers"],
        "gt_source": config["dataset_params"]["gt_source"],
        "gt_type": config["dataset_params"]["gt_type"],
        "gt_dirname": config["dataset_params"]["gt_dirname"],
        "mask_only": mask_only,
    }
    train_val_split_point = config["dataset_params"]["train_fraction"] * 36
    partial_train_tile_filter = functools.partial(inria_aerial_train_tile_filter, train_val_split_point=train_val_split_point)
    partial_val_tile_filter = functools.partial(inria_aerial_val_tile_filter, train_val_split_point=train_val_split_point)

    ds_list = []
    for fold in folds:
        if fold == "train":
            ds = InriaAerial(root_dir, fold="train", tile_filter=partial_train_tile_filter, **kwargs)
            ds_list.append(ds)
        elif fold == "val":
            ds = InriaAerial(root_dir, fold="train", tile_filter=partial_val_tile_filter, **kwargs)
            ds_list.append(ds)
        elif fold == "train_val":
            ds = InriaAerial(root_dir, fold="train", **kwargs)
            ds_list.append(ds)
        elif fold == "test":
            ds = InriaAerial(root_dir, fold="test", **kwargs)
            ds_list.append(ds)
        else:
            print_utils.print_error("ERROR: fold \"{}\" not recognized, implement it in dataset_folds.py.".format(fold))

    return ds_list

''' For large scale real world dataset '''

def private_train_tile_filter(tile, train_val_split_point):
    return tile["number"] <= train_val_split_point # original
    #return int(tile["number"]) <= train_val_split_point # for single image


def private_val_tile_filter(tile, train_val_split_point):
    return train_val_split_point < tile["number"] # original
    #return train_val_split_point < int(tile["number"]) # for single image

def get_private_dataset_folds(config, root_dir, folds):
    from pytorch_lydorn.torch_lydorn.torchvision.datasets.private_dataset import PrivateDataset

    # --- Online transform done on the host (CPU):
    online_cpu_transform = data_transforms.get_online_cpu_transform(config,
                                                                    augmentations=config["data_aug_params"]["enable"])
    #online_cpu_transform = data_transforms.get_online_cpu_transform(config,
                                                #                    augmentations=False) # added by me 19.02
    
    mask_only = config["dataset_params"]["mask_only"]
    kwargs = {
        "pre_process": config["dataset_params"]["pre_process"],
        "transform": online_cpu_transform,
        "patch_size": config["dataset_params"]["data_patch_size"],
        "patch_stride": config["dataset_params"]["input_patch_size"],
        "pre_transform": data_transforms.get_offline_transform_patch(distances=not mask_only, sizes=not mask_only),
        "small": config["dataset_params"]["small"],
        "pool_size": config["num_workers"],
        "gt_source": config["dataset_params"]["gt_source"],
        "gt_type": config["dataset_params"]["gt_type"],
        "gt_dirname": config["dataset_params"]["gt_dirname"],
        "mask_only": mask_only,
    }
    train_val_split_point = config["dataset_params"]["train_fraction"] * 10
    partial_train_tile_filter = functools.partial(private_train_tile_filter, train_val_split_point=train_val_split_point)
    partial_val_tile_filter = functools.partial(private_val_tile_filter, train_val_split_point=train_val_split_point)

    ds_list = []
    for fold in folds:
        if fold == "train":
            ds = PrivateDataset(root_dir, fold="train", tile_filter=partial_train_tile_filter, **kwargs)
            ds_list.append(ds)
        elif fold == "val":
            ds = PrivateDataset(root_dir, fold="train", tile_filter=partial_val_tile_filter, **kwargs)
            ds_list.append(ds)
        elif fold == "train_val":
            ds = PrivateDataset(root_dir, fold="train", **kwargs)
            ds_list.append(ds)
        elif fold == "test":
            ds = PrivateDataset(root_dir, fold="test", **kwargs)
            ds_list.append(ds)
        else:
            print_utils.print_error("ERROR: fold \"{}\" not recognized, implement it in dataset_folds.py.".format(fold))

    return ds_list




def get_folds(config, root_dir, folds):
    assert set(folds).issubset({"train", "val", "train_val", "test"}), \
        'fold in folds should be in ["train", "val", "train_val", "test"]'

    if config["dataset_params"]["root_dirname"] == "AerialImageDataset":
        return get_inria_aerial_folds(config, root_dir, folds)
    
    elif config["dataset_params"]["root_dirname"] == "PrivateDataset":
        return get_private_dataset_folds(config, root_dir, folds)

    else:
        print_utils.print_error("ERROR: config[\"data_root_partial_dirpath\"] = \"{}\" is an unknown dataset! "
                                "If it is a new dataset, add it in dataset_folds.py's get_folds() function.".format(
            config["dataset_params"]["root_dirname"]))
        exit()
