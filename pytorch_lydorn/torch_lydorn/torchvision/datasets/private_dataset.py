import fnmatch
import os.path
import pathlib
import sys
import time

import shapely.geometry
import multiprocess
import itertools
import skimage.io
import numpy as np

from tqdm import tqdm

import cv2

import torch
import torch.utils.data
import torchvision

from lydorn_utils import run_utils, image_utils, polygon_utils, geo_utils
from lydorn_utils import print_utils
from lydorn_utils import python_utils

from pytorch_lydorn.torch_lydorn.torchvision.datasets import utils

CITY_METADATA_DICT = {
    'achim': {
        'fold': 'train',
        'pixelsize': 5, # 10000/2000 = image_size/pixel_size
        "numbers": list(range(1, 10)),
        'mean':[0.4190, 0.4377, 0.4166], 
        'std': [0.1842, 0.1540, 0.1365], 
    },
    'bad_bodenteich': {
        'fold': 'test',
        'pixelsize': 5, # 10000/2000 = image_size/pixel_size
        "numbers": list(range(1, 4)),
        'mean':[0.4190, 0.4377, 0.4166],
        'std': [0.1842, 0.1540, 0.1365], 
    },
    'bremervoerde': {
        'fold': 'test',
        'pixelsize': 5, # 10000/1000 = image_size/pixel_size
        "numbers": list(range(1, 6)),
        'mean':[0.4190, 0.4377, 0.4166],
        'std': [0.1842, 0.1540, 0.1365],  
    },
    'buxtehude': {
        'fold': 'train',
        'pixelsize': 5, # 10000/1000 = image_size/pixel_size
        "numbers": list(range(1, 11)),
        'mean':[0.4190, 0.4377, 0.4166],
        'std': [0.1842, 0.1540, 0.1365], 
    },
    'cuxhaven': {
        'fold': 'train',
        'pixelsize': 5, # 10000/2000 = image_size/pixel_size
        "numbers": list(range(1, 13)),
        'mean':[0.4190, 0.4377, 0.4166],
        'std': [0.1842, 0.1540, 0.1365], 
    },
    'lueneburg': {
        'fold': 'test',
        'pixelsize': 5, # 10000/1000 = image_size/pixel_size
        "numbers": list(range(1, 21)),
        'mean':[0.4190, 0.4377, 0.4166],
        'std': [0.1842, 0.1540, 0.1365],  
    },
    'munster': {
        'fold': 'train',
        'pixelsize': 5, # 10000/1000 = image_size/pixel_size
        "numbers": list(range(1, 8)),
        'mean':[0.4190, 0.4377, 0.4166],
        'std': [0.1842, 0.1540, 0.1365], 
    },
    'nienburg': {
        'fold': 'train',
        'pixelsize': 5, # 10000/2000 = image_size/pixel_size
        "numbers": list(range(1, 17)),
        'mean':[0.4190, 0.4377, 0.4166],
        'std': [0.1842, 0.1540, 0.1365], 
    },
    'visselhoevede': {
        'fold': 'train',
        'pixelsize': 5, # 10000/1000 = image_size/pixel_size
        "numbers": list(range(1, 8)),
        'mean':[0.4190, 0.4377, 0.4166],
        'std': [0.1842, 0.1540, 0.1365], 
    },
    'uelzen': {
        'fold': 'test',
        'pixelsize': 5, # 10000/1000 = image_size/pixel_size
        "numbers": list(range(1, 7)),
        'mean':[0.4190, 0.4377, 0.4166],
        'std': [0.1842, 0.1540, 0.1365],  
    },
}


IMAGE_DIRNAME = "images"
IMAGE_NAME_FORMAT = "{city}{number}"
IMAGE_FILENAME_FORMAT = IMAGE_NAME_FORMAT + ".tif"  # City name, number


class PrivateDataset(torch.utils.data.Dataset):
    """
    Private Image Dataset
    """

    def __init__(self, root: str, fold: str="train", pre_process: bool=True, tile_filter=None, patch_size: int=None, patch_stride: int=None,
                 pre_transform=None, transform=None, small: bool=False, pool_size: int=1, raw_dirname: str="raw", processed_dirname: str="processed",
                 gt_source: str="disk", gt_type: str="npy", gt_dirname: str="gt_polygons", mask_only: bool=False):
        """

        @param root:
        @param fold:
        @param pre_process: If True, the dataset will be pre-processed first, saving training patches on disk. If False, data will be serve on-the-fly without any patching.
        @param tile_filter: Function to call on tile_info, if returns True, include that tile. If returns False, exclude that tile. Does not affect pre-processing.
        @param patch_size:
        @param patch_stride:
        @param pre_transform:
        @param transform:
        @param small: If True, use a small subset of the dataset (for testing)
        @param pool_size:
        @param processed_dirname:
        @param gt_source: Can be "disk" for annotation that are on disk or "osm" to download from OSM (not implemented)
        @param gt_type: Type of annotation files on disk: can be "npy", "geojson" or "tif"
        @param gt_dirname: Name of directory with annotation files
        @param mask_only: If True, discard the RGB image, sample's "image" field is a single-channel binary mask of the polygons and there is no ground truth segmentation.
            This is to allow learning only the frame field from binary masks in order to polygonize binary masks
        """
        assert gt_source in {"disk", "osm"}, "gt_source should be disk or osm"
        assert gt_type in {"npy", "geojson", "tif"}, f"gt_type should be npy, geojson or tif, not {gt_type}"
        self.root = root
        self.fold = fold
        self.pre_process = pre_process
        self.tile_filter = tile_filter
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.pre_transform = pre_transform
        self.transform = transform
        self.small = small
        if self.small:
            print_utils.print_info("INFO: Using small version of the Private dataset.")
        self.pool_size = pool_size
        self.raw_dirname = raw_dirname
        self.gt_source = gt_source
        self.gt_type = gt_type
        self.gt_dirname = gt_dirname
        self.mask_only = mask_only

        # Fill default values
        if self.gt_source == "disk":
            print_utils.print_info("INFO: annotations will be loaded from disk")
        elif self.gt_source == "osm":
            print_utils.print_info("INFO: annotations will be downloaded from OSM. "
                                   "Make sure you have an internet connection to the OSM servers!")

        if self.pre_process:
            # Setup of pre-process
            processed_dirname_extention = f"{processed_dirname}.source_{self.gt_source}.type_{self.gt_type}"
            if self.gt_dirname is not None:
                processed_dirname_extention += f".dirname_{self.gt_dirname}"
            if self.mask_only:
                processed_dirname_extention += f".mask_only_{int(self.mask_only)}"
            processed_dirname_extention += f".patch_size_{int(self.patch_size)}"
            self.processed_dirpath = os.path.join(self.root, processed_dirname_extention, self.fold)
            self.stats_filepath = os.path.join(self.processed_dirpath, "stats-small.pt" if self.small else "stats.pt")
            self.processed_flag_filepath = os.path.join(self.processed_dirpath,
                                                        "processed_flag-small" if self.small else "processed_flag")

            # Check if dataset has finished pre-processing by checking flag:
            if os.path.exists(self.processed_flag_filepath):
                # Process done, load stats
                self.stats = torch.load(self.stats_filepath)
            else:
                # Pre-process not finished, launch it:
                tile_info_list = self.get_tile_info_list(tile_filter=None)
                self.stats = self.process(tile_info_list) # here error
                # Save stats
                torch.save(self.stats, self.stats_filepath)
                # Mark dataset as processed with flag
                pathlib.Path(self.processed_flag_filepath).touch()

            # Get processed_relative_paths with filter
            tile_info_list = self.get_tile_info_list(tile_filter=self.tile_filter)
            self.processed_relative_paths = self.get_processed_relative_paths(tile_info_list)
        else:
            # Setup data sample list
            self.tile_info_list = self.get_tile_info_list(tile_filter=self.tile_filter)

    def get_tile_info_list(self, tile_filter=None):
        tile_info_list = []
        for city, info in CITY_METADATA_DICT.items():
            if not info["fold"] == self.fold:
                continue
            if self.small:
                numbers = [*info["numbers"][:5], info["numbers"][-1]]
            else:
                numbers = info["numbers"]
            for number in numbers:
                image_info = {
                    "city": city,
                    "number": number,
                    "pixelsize": info["pixelsize"],
                    "mean": np.array(info["mean"]),
                    "std": np.array(info["std"]),
                }
                tile_info_list.append(image_info)
        if tile_filter is not None:
            tile_info_list = list(filter(self.tile_filter, tile_info_list))
        return tile_info_list

    def get_processed_relative_paths(self, tile_info_list):
        processed_relative_paths = []
        for tile_info in tile_info_list:
            processed_tile_relative_dirpath = os.path.join(tile_info['city'], f"{tile_info['number']}") # for only one image
            #processed_tile_relative_dirpath = os.path.join(tile_info['city'], f"{tile_info['number']:02d}") # original
            processed_tile_dirpath = os.path.join(self.processed_dirpath, processed_tile_relative_dirpath)
            sample_filenames = fnmatch.filter(os.listdir(processed_tile_dirpath), "data.*.pt")
            processed_tile_relative_paths = [os.path.join(processed_tile_relative_dirpath, sample_filename) for sample_filename
                                        in sample_filenames]
            processed_relative_paths.extend(processed_tile_relative_paths)
        return sorted(processed_relative_paths)

    def process(self, tile_info_list):
        # os.makedirs(os.path.join(self.root, self.processed_dirname), exist_ok=True)
        with multiprocess.Pool(self.pool_size) as p:
            stats_all = list( # here error too
                tqdm(p.imap(self._process_one, tile_info_list), total=len(tile_info_list), desc="Process")) ## Evaluation runs till here then valueerror!

        stats = {}
        if not self.mask_only:
            stats_all = list(filter(None.__ne__, stats_all))
            stat_lists = {}
            for stats_one in stats_all:
                for key, stat in stats_one.items():
                    if key in stat_lists:
                        stat_lists[key].append(stat)
                    else:
                        stat_lists[key] = [stat]

            # Aggregate stats
            if "class_freq" in stat_lists and "num" in stat_lists:
                class_freq_array = np.stack(stat_lists["class_freq"], axis=0)
                num_array = np.stack(stat_lists["num"], axis=0)
                if num_array.min() == 0:
                    raise ZeroDivisionError("num_array has some zeros values, cannot divide!")
                stats["class_freq"] = np.sum(class_freq_array*num_array[:, None], axis=0) / np.sum(num_array)

        return stats

    def load_raw_data(self, tile_info):
        raw_data = {}

        # Image:
        raw_data["image_filepath"] = os.path.join(self.root, self.raw_dirname, self.fold, IMAGE_DIRNAME,
                                      IMAGE_FILENAME_FORMAT.format(city=tile_info["city"], number=tile_info["number"]))
        
        print(raw_data["image_filepath"])
       
        raw_data["image"] = skimage.io.imread(raw_data["image_filepath"])
        #print(raw_data["image"])
        
        '''if len(raw_data["image"].shape) > 2 and raw_data["image"].shape[2] == 4:
            #convert the image from RGBA2RGB
            raw_data["image"] = cv2.cvtColor(raw_data["image"], cv2.COLOR_BGRA2BGR)'''
        
        assert len(raw_data["image"].shape) == 3 and raw_data["image"].shape[2] == 3, f"image should have shape (H, W, 3), not {raw_data['image'].shape}..."

        # Annotations:
        if self.gt_source == "disk":
            gt_base_filepath = os.path.join(self.root, self.raw_dirname, self.fold, self.gt_dirname,
                                            IMAGE_NAME_FORMAT.format(city=tile_info["city"],
                                                                     number=tile_info["number"]))
            gt_filepath = gt_base_filepath + "." + self.gt_type
            if not os.path.exists(gt_filepath):
                raw_data["gt_polygons"] = []
                return raw_data
            if self.gt_type == "npy":
                #print('npy size : ',os.path.getsize(gt_filepath)) # 128 is not zero so the file is not empty!!! then why ran out of input error??
                np_gt_polygons = np.load(gt_filepath, allow_pickle=True) # ndarray
                gt_polygons = []
                for np_gt_polygon in np_gt_polygons:
                    try:
                        gt_polygons.append(shapely.geometry.Polygon(np_gt_polygon[:, ::-1]))
                    except ValueError:
                        # Invalid polygon, continue without it
                        continue
                raw_data["gt_polygons"] = gt_polygons
            elif self.gt_type == "geojson":
                geojson = python_utils.load_json(gt_filepath)
                raw_data["gt_polygons"] = list(shapely.geometry.shape(geojson))
            elif self.gt_type == "tif":
                raw_data["gt_polygons_image"] = skimage.io.imread(gt_filepath)[:, :, None]
                assert len(raw_data["gt_polygons_image"].shape) == 3 and raw_data["gt_polygons_image"].shape[2] == 1, \
                    f"Mask should have shape (H, W, 1), not {raw_data['gt_polygons_image'].shape}..."
        elif self.gt_source == "osm":
            raise NotImplementedError(
                "Downloading from OSM is not implemented (takes too long to download, better download to disk first...).")
            # np_gt_polygons = geo_utils.get_polygons_from_osm(image_filepath, tag="building", ij_coords=False)

        #print(raw_data)
        return raw_data

    def _process_one(self, tile_info):
        process_id = int(multiprocess.current_process().name[-1])
        # print(f"\n--- {process_id} ---\n")

        # --- Init
        tile_name = IMAGE_NAME_FORMAT.format(city=tile_info["city"], number=tile_info["number"])
        processed_tile_relative_dirpath = os.path.join(tile_info['city'], f"{tile_info['number']}") # when only one image is used!
        #processed_tile_relative_dirpath = os.path.join(tile_info['city'], f"{tile_info['number']:02d}") # original line
        processed_tile_dirpath = os.path.join(self.processed_dirpath, processed_tile_relative_dirpath)
        processed_flag_filepath = os.path.join(processed_tile_dirpath, "processed_flag")
        stats_filepath = os.path.join(processed_tile_dirpath, "stats.pt")
        os.makedirs(processed_tile_dirpath, exist_ok=True)
        stats = {}

        # --- Check if tile has been processed already
        if os.path.exists(processed_flag_filepath):
            if not self.mask_only:
                stats = torch.load(stats_filepath)
            return stats

        # --- Read data:
        raw_data = self.load_raw_data(tile_info)
        #print(raw_data) # added by me 02.03.2022

        # --- Patch tiles
        if self.patch_size is not None:
            patch_stride = self.patch_stride if self.patch_stride is not None else self.patch_size
            patch_boundingboxes = image_utils.compute_patch_boundingboxes(raw_data["image"].shape[0:2],
                                                                          stride=patch_stride,
                                                                          patch_res=self.patch_size)
            class_freq_list = []
            for i, bbox in enumerate(tqdm(patch_boundingboxes, desc=f"Patching {tile_name}", leave=False, position=process_id)): ## private_dataset.py gets till here
                sample = {
                    "image_filepath": raw_data["image_filepath"],
                    "name": f"{tile_name}.rowmin_{bbox[0]}_colmin_{bbox[1]}_rowmax_{bbox[2]}_colmax_{bbox[3]}",
                    "bbox": bbox,
                    "city": tile_info["city"],
                    "number": tile_info["number"],
                }

                if self.gt_type == "npy" or self.gt_type == "geojson":
                    patch_gt_polygons = polygon_utils.patch_polygons(raw_data["gt_polygons"], minx=bbox[1], miny=bbox[0],
                                                                     maxx=bbox[3], maxy=bbox[2])
                    sample["gt_polygons"] = patch_gt_polygons
                elif self.gt_type == "tif":
                    patch_gt_mask = raw_data["gt_polygons_image"][bbox[0]:bbox[2], bbox[1]:bbox[3], :]
                    sample["gt_polygons_image"] = patch_gt_mask

                sample["image"] = raw_data["image"][bbox[0]:bbox[2], bbox[1]:bbox[3], :]
                #print(len(sample["image"]))

                sample = self.pre_transform(sample)  # Needs "image" to infer shape even if mask_only is True # gets stuck here
                if self.mask_only:
                    del sample["image"]  # Don't need RGB image anymore

                relative_filepath = os.path.join(processed_tile_relative_dirpath, "data.{:06d}.pt".format(i))
                filepath = os.path.join(self.processed_dirpath, relative_filepath)
                torch.save(sample, filepath)
                #print('Sample line 365 : ',sample)

                # Compute stats
                if not self.mask_only:
                    if self.gt_type == "npy" or self.gt_type == "geojson":
                        class_freq_list.append(np.mean(sample["gt_polygons_image"], axis=(0, 1)) / 255)
                    elif self.gt_type == "mask":
                        raise NotImplementedError("mask class freq")
                    else:
                        raise NotImplementedError(f"gt_type={self.gt_type} not implemented for computing stats")

            # Aggregate stats
            if not self.mask_only:
                if len(class_freq_list):
                    class_freq_array = np.stack(class_freq_list, axis=0)
                    stats["class_freq"] = np.mean(class_freq_array, axis=0)
                    stats["num"] = len(class_freq_list)
                else:
                    print("Empty tile:", tile_info["city"], tile_info["number"], "polygons:", len(raw_data["gt_polygons"]))
        else:
            raise NotImplemented("patch_size is None")

        # Save stats
        if not self.mask_only:
            torch.save(stats, stats_filepath)

        # Mark tile as processed with flag
        pathlib.Path(processed_flag_filepath).touch()

        return stats

    def __len__(self):
        if self.pre_process:
            return len(self.processed_relative_paths)
        else:
            return len(self.tile_info_list)

    def __getitem__(self, idx):
        if self.pre_process:
            filepath = os.path.join(self.processed_dirpath, self.processed_relative_paths[idx])
            data = torch.load(filepath)
            if self.mask_only:
                data["image"] = np.repeat(data["gt_polygons_image"][:, :, 0:1], 3, axis=-1)  # Fill image slot
                data["image_mean"] = np.array([0.5, 0.5, 0.5])
                data["image_std"] = np.array([1, 1, 1])
            else:
                data["image_mean"] = np.array(CITY_METADATA_DICT[data["city"]]["mean"])
                data["image_std"] = np.array(CITY_METADATA_DICT[data["city"]]["std"])
                data["class_freq"] = self.stats["class_freq"]
        else:
            tile_info = self.tile_info_list[idx]
            # Load raw data
            data = self.load_raw_data(tile_info)
            data["name"] = IMAGE_NAME_FORMAT.format(city=tile_info["city"], number=tile_info["number"])
            data["image_mean"] = np.array(tile_info["mean"])
            data["image_std"] = np.array(tile_info["std"])
            
        data = self.transform(data)
        return data


def main():
    # Test using transforms from the frame_field_learning project:
    from frame_field_learning import data_transforms

    config = {
        "data_dir_candidates": [
            '~/Polygonization-by-Frame-Field-Learning/data'
        ],
        "dataset_params": {
            "root_dirname": "PrivateDataset",
            "pre_process": True,
            "gt_source": "disk",
            "gt_type": "tif", # tif
            "gt_dirname": "gt", # gt
            "mask_only": False,
            "small": False,
            "data_patch_size": 425,
            "input_patch_size": 300,

            "train_fraction": 0.75
        },
        "num_workers": 0,
        
        "data_aug_params": {
            "enable": True,
            "vflip": True,
            "affine": True,
            "scaling": [0.9, 1.1],
            "color_jitter": True,
            "device": "cuda"
        }
    }

    # Find data_dir
    data_dir = python_utils.choose_first_existing_path(config["data_dir_candidates"])
    if data_dir is None:
        print_utils.print_error("ERROR: Data directory not found!")
        exit()
    else:
        print_utils.print_info("Using data from {}".format(data_dir))
    root_dir = os.path.join(data_dir, config["dataset_params"]["root_dirname"])

    # --- Transforms: --- #
    # --- pre-processing transform (done once then saved on disk):
    # --- Online transform done on the host (CPU):
    online_cpu_transform = data_transforms.get_online_cpu_transform(config,
                                                                    augmentations=config["data_aug_params"]["enable"])
    train_online_cuda_transform = data_transforms.get_online_cuda_transform(config, augmentations=config["data_aug_params"]["enable"])
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
        "mask_only": config["dataset_params"]["mask_only"],
    }
    train_val_split_point = config["dataset_params"]["train_fraction"] * 10
    #def train_tile_filter(tile): return tile["number"] <= train_val_split_point # original
    #def val_tile_filter(tile): return train_val_split_point < tile["number"] # original
    def train_tile_filter(tile): return int(tile["number"]) <= train_val_split_point # for single image
    def val_tile_filter(tile): return train_val_split_point < int(tile["number"]) # for single image
    
    # --- --- #
    fold = "train"
    if fold == "train":
        dataset = PrivateDataset(root_dir, fold="train", tile_filter=train_tile_filter, **kwargs)
    elif fold == "val":
        dataset = PrivateDataset(root_dir, fold="train", tile_filter=val_tile_filter, **kwargs)
    elif fold == "test":
        dataset = PrivateDataset(root_dir, fold="test", **kwargs)

    print(f"dataset has {len(dataset)} samples.")
    
    print("# --- Sample 0 --- #")
    sample = dataset[0]
    for key, item in sample.items():
        print("{}: {}".format(key, type(item)))

    print("# --- Samples --- #")
    # for data in tqdm(dataset):
    #     pass

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config["num_workers"])
    print("# --- Batches --- #")
    for batch in tqdm(data_loader):

        # batch["distances"] = batch["distances"].float()
        # batch["sizes"] = batch["sizes"].float()

        # im = np.array(batch["image"][0])
        # im = np.moveaxis(im, 0, -1)
        # skimage.io.imsave('im_before_transform.png', im)
        #
        # distances = np.array(batch["distances"][0])
        # distances = np.moveaxis(distances, 0, -1)
        # skimage.io.imsave('distances_before_transform.png', distances)
        #
        # sizes = np.array(batch["sizes"][0])
        # sizes = np.moveaxis(sizes, 0, -1)
        # skimage.io.imsave('sizes_before_transform.png', sizes)

        print("----")
        print(batch["name"])

        print("image:", batch["image"].shape, batch["image"].min().item(), batch["image"].max().item())
        im = np.array(batch["image"][0])
        im = np.moveaxis(im, 0, -1)
        skimage.io.imsave('im.png', im)

        if "gt_polygons_image" in batch:
            print("gt_polygons_image:", batch["gt_polygons_image"].shape, batch["gt_polygons_image"].min().item(),
                  batch["gt_polygons_image"].max().item())
            seg = np.array(batch["gt_polygons_image"][0]) / 255
            seg = np.moveaxis(seg, 0, -1)
            seg_display = utils.get_seg_display(seg)
            seg_display = (seg_display * 255).astype(np.uint8)
            skimage.io.imsave("gt_seg.png", seg_display)

        if "gt_crossfield_angle" in batch:
            print("gt_crossfield_angle:", batch["gt_crossfield_angle"].shape, batch["gt_crossfield_angle"].min().item(),
                  batch["gt_crossfield_angle"].max().item())
            gt_crossfield_angle = np.array(batch["gt_crossfield_angle"][0])
            gt_crossfield_angle = np.moveaxis(gt_crossfield_angle, 0, -1)
            skimage.io.imsave('gt_crossfield_angle.png', gt_crossfield_angle)

        if "distances" in batch:
            print("distances:", batch["distances"].shape, batch["distances"].min().item(), batch["distances"].max().item())
            distances = np.array(batch["distances"][0])
            distances = np.moveaxis(distances, 0, -1)
            skimage.io.imsave('distances.png', distances)

        if "sizes" in batch:
            print("sizes:", batch["sizes"].shape, batch["sizes"].min().item(), batch["sizes"].max().item())
            sizes = np.array(batch["sizes"][0])
            sizes = np.moveaxis(sizes, 0, -1)
            skimage.io.imsave('sizes.png', sizes)

        # valid_mask = np.array(batch["valid_mask"][0])
        # valid_mask = np.moveaxis(valid_mask, 0, -1)
        # skimage.io.imsave('valid_mask.png', valid_mask)

        print("Apply online tranform:")
        batch = utils.batch_to_cuda(batch)
        batch = train_online_cuda_transform(batch)
        batch = utils.batch_to_cpu(batch)

        print("image:", batch["image"].shape, batch["image"].min().item(), batch["image"].max().item())
        print("gt_polygons_image:", batch["gt_polygons_image"].shape, batch["gt_polygons_image"].min().item(), batch["gt_polygons_image"].max().item())
        print("gt_crossfield_angle:", batch["gt_crossfield_angle"].shape, batch["gt_crossfield_angle"].min().item(), batch["gt_crossfield_angle"].max().item())
        # print("distances:", batch["distances"].shape, batch["distances"].min().item(), batch["distances"].max().item())
        # print("sizes:", batch["sizes"].shape, batch["sizes"].min().item(), batch["sizes"].max().item())

        # Save output to visualize
        seg = np.array(batch["gt_polygons_image"][0])
        seg = np.moveaxis(seg, 0, -1)
        seg_display = utils.get_seg_display(seg)
        seg_display = (seg_display * 255).astype(np.uint8)
        skimage.io.imsave("gt_seg.png", seg_display)

        im = np.array(batch["image"][0])
        im = np.moveaxis(im, 0, -1)
        skimage.io.imsave('im.png', im)

        gt_crossfield_angle = np.array(batch["gt_crossfield_angle"][0])
        gt_crossfield_angle = np.moveaxis(gt_crossfield_angle, 0, -1)
        skimage.io.imsave('gt_crossfield_angle.png', gt_crossfield_angle)

        distances = np.array(batch["distances"][0])
        distances = np.moveaxis(distances, 0, -1)
        skimage.io.imsave('distances.png', distances)

        sizes = np.array(batch["sizes"][0])
        sizes = np.moveaxis(sizes, 0, -1)
        skimage.io.imsave('sizes.png', sizes)

        # valid_mask = np.array(batch["valid_mask"][0])
        # valid_mask = np.moveaxis(valid_mask, 0, -1)
        # skimage.io.imsave('valid_mask.png', valid_mask)

        input("Press enter to continue...")


if __name__ == '__main__':
    main()
