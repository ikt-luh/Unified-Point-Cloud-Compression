import torch
import random
import math
import MinkowskiEngine as ME

class Q_Map(object):
    """
    Generator for Quality maps
    """
    def __init__(self, config):
        self.mode = config["mode"]
        if self.mode == "exponential":
            self.a_A = math.log2(config["lambda_A_max"] + config["lambda_A_min"])
            self.b_A = config["lambda_A_min"] - 1
            self.a_G = math.log2(config["lambda_G_max"] + config["lambda_G_min"])
            self.b_G = config["lambda_G_min"] - 1
        elif self.mode == "quadratic":
            self.a_A = config["lambda_A_max"] - config["lambda_A_min"]
            self.b_A = config["lambda_A_min"]
            self.a_G = config["lambda_G_max"] - config["lambda_G_min"]
            self.b_G = config["lambda_G_min"]


    def __call__(self, geometry):
        """
        Documentation
        
        Parameters
        ----------
        geometry: ME.SparseTensor
            Description
        
        returns
        -------
        q_map: ME.SparseTensor
            Q_Map of the data
        """
        batch_indices = torch.unique(geometry.C[:, 0])

        q_map = torch.zeros((torch.max(batch_indices)+1, 2), device=geometry.device)
        q_map[:, 0] = random.uniform(0, 1)
        q_map[:, 1] = random.uniform(0, 1)

        # Scale 
        lambda_map = self.scale_q_map(q_map)
        return q_map, lambda_map


    def scale_q_map(self, q_map):
        """
        Scales the q_map to receive a Lambda map for loss computation
        """
        lambda_map_features = q_map.clone()
        if self.mode == "exponential":
            lambda_map_features[:, 0] = 2**(lambda_map_features[:, 0] * self.a_G) + self.b_G
            lambda_map_features[:, 1] = 2**(lambda_map_features[:, 1] * self.a_A) + self.b_A
        elif self.mode == "quadratic":
            lambda_map_features[:, 0] = lambda_map_features[:, 0]**2 * self.a_G + self.b_G
            lambda_map_features[:, 1] = lambda_map_features[:, 1]**2 * self.a_A + self.b_A
        else:
            raise ValueError("Unknown Q_map mode")
        return lambda_map_features