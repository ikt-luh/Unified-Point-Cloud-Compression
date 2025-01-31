import torch
import random
import math
import MinkowskiEngine as ME

class Q_Func(object):
    """
    Generator for Quality values and lambdas
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
        Generate the quality map for a point cloud
        
        Parameters:
            geometry (ME.SparseTensor):
                Input geometry
        
        Returns:
            q_vals (torch.tensor):
                q_values for the data
            lambda_vals (torch.tensor):
                lambda_values for the data
        """
        batch_indices = torch.unique(geometry.C[:, 0])

        q_vals = torch.zeros((torch.max(batch_indices)+1, 2), device=geometry.device)
        q_vals[:, 0] = random.uniform(0, 1)
        q_vals[:, 1] = random.uniform(0, 1)

        # Scale 
        lambda_map = self.scale_q_vals(q_vals)
        return q_vals, lambda_map


    def scale_q_vals(self, q_vals):
        """
        Scales the q_vals to receive a Lambda map for loss computation using the 
        scaling function from the config.

        Parameters:
            q_vals (torch.tensor):
                Quality values

        Returns:
            lambda_vals (torch.tensor)
                Corresponding lambda values
        """
        lambda_vals = q_vals.clone()
        if self.mode == "exponential":
            lambda_vals[:, 0] = 2**(lambda_vals[:, 0] * self.a_G) + self.b_G
            lambda_vals[:, 1] = 2**(lambda_vals[:, 1] * self.a_A) + self.b_A
        elif self.mode == "quadratic":
            lambda_vals[:, 0] = lambda_vals[:, 0]**2 * self.a_G + self.b_G
            lambda_vals[:, 1] = lambda_vals[:, 1]**2 * self.a_A + self.b_A
        else:
            raise ValueError("Unknown mapping mode")
        return lambda_vals