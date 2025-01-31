import os
import torch
import random
import torchvision
import numpy as np
import MinkowskiEngine as ME


def build_transforms(config):
    """
    Wrapper class for building transforms
    """
    transforms = []
    if not config:
        return transforms
    
    config = {k: config[k] for k in sorted(config)}

    for name, setting in config.items():
        key = setting["key"]

        if key == "RandomRotate":
            block_size = setting["block_size"]
            transforms.append(RandomRotate(block_size))
        elif key == "ColorJitter":
            transforms.append(ColorJitter())
        else:
            raise ValueError("Transform {key} not defined.")
    
    return transforms

class ColorJitter(object):
    """
    Randomly Jitters the color (wraps ColorJitter from Torch)
    """
    def __init__(self):
        self.jitter = torchvision.transforms.ColorJitter(brightness=0.3,
                                                         contrast=0.3,
                                                         saturation=0.3,
                                                         hue=0.3)

    def __call__(self, sample):
        if "cubes" in sample:
            for idx, cube in enumerate(sample["cubes"]):
                sample["cubes"][idx] = self.transform(cube)
            return sample
        else:
            return self.transform(sample)
        
    def transform(self, sample):
        sample["colors"] = self.jitter(sample["colors"].T.unsqueeze(-1))
        sample["colors"] = sample["colors"].squeeze(-1).T

        return sample


class RandomRotate(object):
    """ 
    Randomly rotate the point cloud in 3D using PyTorch, 
    filter out-of-bounds points, round them back integers, and remove duplicates.
    """
    def __init__(self, block_size, crop=False):
        self.block_size = block_size
        self.crop = crop

    def __call__(self, sample):
        if "cubes" in sample:
            for idx, cube in enumerate(sample["cubes"]):
                sample["cubes"][idx] = self.transform(cube)
            return sample
        else:
            return self.transform(sample)

    def transform(self, sample):
        phi = torch.rand(1) * 2 * 3.141592653589793  # Random roll
        theta = torch.rand(1) * 2 * 3.141592653589793  # Random pitch

        rotation_matrix = self.rotation_matrix_3d(phi, theta)
        points = sample["points"].clone()
        colors = sample["colors"].clone()

        # Applying rotation to the points
        rotated_points = torch.mm(points - self.block_size/2, rotation_matrix.T)
        rotated_points = rotated_points + self.block_size/2

        if self.crop:
            valid_indices = ((rotated_points >= 0) & (rotated_points < self.block_size)).all(dim=1)
            valid_points = rotated_points[valid_indices]
            valid_colors = colors[valid_indices]

        else:
            valid_points = rotated_points
            valid_colors = colors

        # Remove duplicate points
        unique_points, unique_colors = ME.utils.sparse_quantize(
                coordinates=valid_points,
                features=valid_colors,
                quantization_size=1.0
            )

        sample['points'] = unique_points
        sample['colors'] = unique_colors

        return sample
    

    @staticmethod
    def rotation_matrix_3d(phi, theta):
        """
        Generate a random 3D rotation matrix 
        """
        
        R_x = torch.tensor([[1, 0, 0],
                            [0, torch.cos(phi), -torch.sin(phi)],
                            [0, torch.sin(phi), torch.cos(phi)]])
        
        R_y = torch.tensor([[torch.cos(theta), 0, torch.sin(theta)],
                            [0, 1, 0],
                            [-torch.sin(theta), 0, torch.cos(theta)]])
        
        R = torch.mm(R_y, R_x)
        return R
