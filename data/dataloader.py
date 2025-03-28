import os
import yaml
import torch
import open3d as o3d
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
from copy import deepcopy

from .utils.RawLoader import RawLoader


class StaticDataset():
    """ Static Point Cloud dataset """

    def __init__(self, data_dir, split, type="voxels", partition=True, min_points=0, transform=None):
        """
        Parameters:
            root_dir (str):
                Root directory of the dataset
            split (str):
                Split of the data, options: train, test, val
            type (str):
                Type of the data: [voxels, points]
            pointclouds (boolean):
                Load full pointclouds, else just cubes
            min_points (int):
                Minimum points to be loaded, only applies to train split
            transform (torchvision.transforms.Compose):
                Object holding the data transformations, default: None
        """
        self.data_dir = data_dir

        self.split = split
        self.type = type
        self.partition = partition

        self.min_points = min_points
        self.transform = transform

        if split not in ["train", "test", "val"]:
            raise ValueError("Split not defined")

        self.load_data()



    def __len__(self):
        return len(self.indices)


    def __getitem__(self, idx):
        """
        Retrieves an item for the dataloader
        """
        if self.partition:
            (sequence, frameIdx, cubeIdx) = self.indices[idx]
            sample = self.data[sequence][frameIdx]["cubes"][cubeIdx]
        else:
            (sequence, frameIdx) = self.indices[idx]
            sample = self.data[sequence][frameIdx]

        # Create a copy
        sample = deepcopy(sample)

        if self.transform:
            sample = self.transform(sample)
        return sample

    @staticmethod
    def collate_fn(samples):
        """
        Function to handle batching.
        dataset = CubeDataset(data_list)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=CubeDataset.collate_fn)
        """
        pass


    def load_data(self):
        """
        Load the data into RAM. If not existant, prepare the dataset.
        """
        data_path = os.path.join(self.data_dir, "{}.pt".format(self.split))
        if not os.path.exists(data_path):
            self.prepare_split(data_path)

        print("Loading data from {}".format(data_path))
        self.data = torch.load(data_path)
        print("Dataloader ready!")

        # Prepare indices
        self.prepare_indices()


    def prepare_split(self, data_path):
        """
        Prepare the split

        Parameters:
            data_path (str):
                Path to the target dataset directory
        """
        config_path = os.path.join(self.data_dir, "config.yaml")
        config = self.parse_config(config_path)
        split_config = config[self.split]

        current_directory = os.path.dirname(os.path.abspath((__file__)))
        raw_data_config = os.path.join(current_directory, "config", "raw_loading.yaml")
        raw_data_path = os.path.join(current_directory, "datasets", "raw")
        raw_loader = RawLoader(raw_data_path, raw_data_config)

        data = {}
        for sequence, frames in split_config.items():
            data[sequence] = {}
            for frame in frames:
                data[sequence][frame] = {}

        #Iterate through .ply objects and slice them
        for sequence, frames in split_config.items():
            for frameIdx in frames:
                t0 = time.time()
                orig_pointcloud = raw_loader.get_pointcloud(sequence, frameIdx)
                points = torch.from_numpy(np.asarray(orig_pointcloud.points)).float()
                colors = torch.from_numpy(np.asarray(orig_pointcloud.colors)).float()
                t_delta = time.time() - t0
                print("Loaded {}_{} in {:0.3f}s".format(sequence, str(frameIdx), t_delta))
                
                # Move to cuda for faster slicing
                t0 = time.time()
                if torch.cuda.is_available():
                    points.cuda()
                    colors.cuda()

                # Slice into cubes
                cubes = self.slice_into_cubes(points, colors, config["info"]["cube_size"])

                # and back to CPU
                if torch.cuda.is_available():
                    points.cpu()
                    colors.cpu()
                    for cube in cubes:
                        for _, cube_data in cube.items():
                            cube_data.cpu()

                t_delta = time.time() - t0
                print("Sliced {}_{} in {:0.3f}s ({} cubes)".format(sequence, str(frameIdx), t_delta, len(cubes)))

                # Add frame and sequence info
                for cube in cubes:
                    cube["frameIdx"] = frameIdx
                    cube["sequence"] = sequence
                    cube["blocks"] = config["info"]["cube_size"]

                # Build the data 
                data[sequence][frameIdx]["cubes"] = cubes
                data[sequence][frameIdx]["src"] = {
                    "points": points.detach(),
                    "colors": colors.detach()
                }

        t0 = time.time()
        print("Slicing done, saving to disk")
        torch.save(data, data_path)
        print("Saved in {:0.3f}s".format(time.time() - t0))


    def slice_into_cubes(self, points_tensor, colors_tensor, cube_size=64):
        """
        Slice a pointcloud into cubes.

        Parameters:
            points_tensor (torch.tensor):
                Point locations
            color_tensor (torch.tensor):
                Point colors
            cube_size (int, default=64):
                Target size of the cubes
        
        Returns:
            cubes (list):
                List of all cubes
                Each cube is a dictionary with keys points, colors, offset, num_points
        """
        min_boundary = torch.tensor([0,0,0])
        cube_indices = ((points_tensor - min_boundary) / cube_size).floor().long()
        unique_cube_indices, inverse_indices = torch.unique(cube_indices, dim=0, return_inverse=True)

        cubes = []
        for idx in range(unique_cube_indices.size(0)):
            mask = inverse_indices == idx

            cube_points = points_tensor[mask]
            cube_colors = colors_tensor[mask]

            cube_shift = unique_cube_indices[idx] * cube_size
            adjusted_points = cube_points - cube_shift

            cube = {
                "points": adjusted_points.detach(),
                "colors": cube_colors.detach(),
                "offset": cube_shift.detach(),
                "num_points": torch.tensor(len(adjusted_points)).detach()
            }
            if cube["num_points"] > 0:
                cubes.append(cube)

        return cubes


    def prepare_indices(self):
        """
        Prepare indices for computing length and accessing data
        """
        self.indices = []
        if self.partition:
            # Cube partition
            for sequence, frames in self.data.items():
                for frameIdx in frames:
                    cubes = self.data[sequence][frameIdx]["cubes"]
                    cubeIdx = [i for i, cube in enumerate(cubes) if cube["num_points"] > self.min_points]
                    index = [(sequence, frameIdx, i) for i in cubeIdx]
                    self.indices.extend(index)
        else: 
            # Point Cloud partition
            for sequence, frames in self.data.items():
                for frameIdx in frames:
                    index = (sequence, frameIdx)
                    self.indices.append(index)
    
    
    def parse_config(self, config_path):
        """
        Parse the config to a dict with datasets, sequences and list of frames.

        Parameters:
            config_path (str):
                Path to the configuration

        Returns:
            config (dict):
                Config dictionary
        """
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        for split , sub_dicts in config.items():
            if split == "info":
                continue # Skip info

            for key, item in sub_dicts.items():
                new_item = []
                if not isinstance(item, str):
                    raise ValueError("Cannot parse config, all keys should be str.")

                sub_items = item.split(",")
                for sub_item in sub_items:
                    # Hanlde start, end, (stride=1) notation
                    if ":" in sub_item:
                        elements = sub_item.split(":")
                        if len(elements) == 2:
                            stride = 1
                        elif len(elements) == 3:
                            stride = int(elements[2])

                        sub_range = list(range(int(elements[0]), int(elements[1])+1, stride))
                        new_item += sub_range

                    # Handle single indexing
                    elif isinstance(sub_item, str):
                        new_item.append(int(sub_item))

                new_item.sort()
                new_item = list(set(new_item)) # Removes redundancies
                config[split][key] = new_item

        return config