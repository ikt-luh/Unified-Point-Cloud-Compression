import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import math
import MinkowskiEngine as ME


class Loss():
    """
    Wrapper holding loss functions
    """
    def __init__(self, config):
        """
        Parameters:
            config (dict):
                Dictionary containing loss configurations
        """
        self.losses = {}
        for id, setting in config.items():
            key = setting["type"]
            setting["id"] = id

            # Match key to the respective class
            match key:
                case "BPPLoss":
                    self.losses[id] = BPPLoss(setting)
                case "ColorLoss":
                    self.losses[id] = ColorLoss(setting)
                case "ShepardsLoss":
                    self.losses[id] = ShepardsLoss(setting)
                case "Multiscale_FocalLoss":
                    self.losses[id] = Multiscale_FocalLoss(setting)
                case _:
                    print("Not found {}".format(key))
                
                    
    def __call__(self, gt, pred):
        """
        Call the loss function to return sum of all losses
        
        Parameters:
            gt (ME.SparseTensor):
                Ground truth point cloud
            pred (dict):
                Dictionary containing information for computing the loss

        Returns:
            total_loss (torch.tensor):
                Total loss after adding and weighting
            losses (dict):
                Dictionary containing the loss value per loss
        """
        total_loss = 0
        losses = {}
        for _, loss in self.losses.items():
            loss_item = loss(gt, pred)
            losses[loss.identifier] = loss_item
            total_loss += loss_item
        
        return total_loss, losses
        
class BPPLoss():
    """
    BPP loss
    """
    def __init__(self, config):
        self.weight = config["weight"]
        self.identifier = config["id"]
        self.key = config["key"]

    def __call__(self, gt, pred):
        loss = 0.0
        likelihoods = pred["likelihoods"][self.key]
        num_points = gt.C.shape[0]

        for likelihood in likelihoods:
            bits = torch.log(likelihood).sum() / (- math.log(2) * num_points)
            loss += bits

        return loss.mean() * self.weight
    

class ColorLoss():
    """
    ColorLoss using L2/L1 on GT voxel locations
    """
    def __init__(self, config):
        self.identifier = config["id"]
        if config["loss"] == "L1":
            self.loss_func = torch.nn.L1Loss(reduction="none")
        elif config["loss"] == "L2":
            self.loss_func = torch.nn.MSELoss(reduction="none")

    def __call__(self, gt, pred):
        prediction = pred["prediction"]
        q_map = pred["q_map"]

        scaling_factors = torch.tensor([1, 1e6, 1e12, 1e18], dtype=torch.int64, device=gt.C.device)
        gt_flat = (gt.C.to(torch.int64) * scaling_factors).sum(dim=1)
        pred_flat = (prediction.C.to(torch.int64) * scaling_factors).sum(dim=1)

        overlapping_mask = torch.isin(gt_flat, pred_flat)

        pred_colors = prediction.features_at_coordinates(gt.C[overlapping_mask].float())
        gt_colors = gt.F[overlapping_mask]

        batch_mask = gt.C[overlapping_mask, 0]
        color_loss = self.loss_func(gt_colors, pred_colors) * q_map[batch_mask, 1].unsqueeze(1)

        return color_loss.mean()
    


class Multiscale_FocalLoss():
    """
    Focal Loss for predicted voxels
    for gamma = 0, this is BCE
    """
    def __init__(self, config):
        self.identifier = config["id"]
        self.alpha = config["alpha"]
        self.gamma = config["gamma"]
        self.pooling = ME.MinkowskiAvgPooling(kernel_size=3, stride=1, dimension=3)
        self.down_pooling = ME.MinkowskiAvgPooling(kernel_size=3, stride=2, dimension=3)

    def __call__(self, gt, pred):
        predictions = pred["occ_predictions"]
        points = pred["points"]
        predictions.reverse()
        points.reverse()

        q_map = pred["q_map"]


        loss = 0.0
        for prediction, coords in zip(predictions, points):
            scaling_factors = torch.tensor([1, 1e4, 1e9, 1e14], dtype=torch.int64, device=gt.C.device)
            gt_flat = (coords.C.to(torch.int64) * scaling_factors).sum(dim=1)
            pred_flat = (prediction.C.to(torch.int64) * scaling_factors).sum(dim=1)

            overlapping_mask = torch.isin(pred_flat, gt_flat)
        
            p_z = F.sigmoid(prediction.F[:, 0]) # F[0] contains occupancy
        
            # Build pt_z and alpha_z
            pt_z = torch.where(overlapping_mask, p_z, 1 - p_z)
            alpha_z = torch.where(overlapping_mask, self.alpha, 1 - self.alpha)
            pt_z = torch.clip(pt_z, 1e-2, 1)

            # Focal Loss
            focal_loss = - alpha_z * (1-pt_z)**self.gamma * torch.log(pt_z)
            
            batch_mask = prediction.C[:, 0]
            loss += (focal_loss * q_map[batch_mask, 0]).mean()

        return loss



class ShepardsLoss():
    """
    Shepard's Loss, using L2/L1 color loss on ground truth voxel locations. (Used in the ablation)
    """
    def __init__(self, config):
        self.identifier = config["id"]

        # Choose loss function based on configuration
        if config["loss"] == "L1":
            self.loss_func = torch.nn.L1Loss(reduction="none")
        elif config["loss"] == "L2":
            self.loss_func = torch.nn.MSELoss(reduction="none")

        self.p = config["p"]
        self.window_size = config["window_size"]

        self.window = self.create_window_3D(self.window_size)

        self.conv_sum = self._init_minkowski_conv(4, self.window_size, self.window)

    def _init_minkowski_conv(self, in_channels, kernel_size, kernel):
        """
        Initialize a Minkowski Channelwise Convolution and set its kernel.
        """
        conv = ME.MinkowskiChannelwiseConvolution(in_channels=in_channels, kernel_size=kernel_size, stride=1, dimension=3)
        conv.kernel = torch.nn.Parameter(kernel)
        conv.kernel.requires_grad = False
        return conv


    def create_window_3D(self, window_size):
        """
        Compute a 3D window shaped like a ball with inverse distance weighting.

        Parameters:
            window_size (int):
                Size of the window (must be odd for a symmetric ball)

        Returns:
            window (torch.tensor):
                3D window of shape (window_size**3) with inverse distance weighting
        """
        radius = window_size // 2
        window = torch.zeros((window_size, window_size, window_size))

        z, y, x = torch.meshgrid(
            torch.arange(window_size) - radius,
            torch.arange(window_size) - radius,
            torch.arange(window_size) - radius
        )

        distance = torch.sqrt(x**2 + y**2 + z**2)
        window = 1 / (distance ** self.p + 1e-5)
        window[distance > radius] = 0

        window = window.view(-1, 1)
        return window

    def __call__(self, gt, pred):
        self.conv_sum.to(gt.C.device)
        prediction = ME.SparseTensor(coordinates=pred["prediction"].C, features=pred["prediction"].F, device=pred["prediction"].C.device)
        q_map = pred["q_map"]

        gt_on_pred = self.interpolate_gt_to_pred(gt, prediction)

        valid_mask = (~torch.isnan(gt_on_pred.F) & ~torch.isinf(gt_on_pred.F)).all(dim=1)
        batch_indicies = gt_on_pred.C[valid_mask, 0]
        color_loss = self.loss_func(gt_on_pred.F[valid_mask], prediction.F[valid_mask]) * q_map[batch_indicies, 1].unsqueeze(1)
        return color_loss.mean()


    def interpolate_gt_to_pred(self, gt, prediction, interpolate_q_map=False):
        """
        Interpolate ground truth values to predicted coordinates using Minkowski convolution.
        """      
        N = 3 if interpolate_q_map else 4

        # Duplicate removal (Some rare cases)
        overlapping_mask = utils.overlapping_mask(prediction, gt)

        # Concatenate gt and non-overlapping prediction coordinates
        combined_coords = torch.cat([gt.C, prediction.C[~overlapping_mask]])
        combined_tensor = ME.SparseTensor(
            coordinates=combined_coords,
            features=torch.ones(combined_coords.shape[0], N, device=gt.C.device),
            device=gt.C.device
        )

        overlapping_mask_comb = utils.overlapping_mask(combined_tensor, gt)

        # Update the features for overlapping and non-overlapping points
        combined_tensor.F[~overlapping_mask_comb] = 0.0
        combined_tensor.F[overlapping_mask_comb, 1] = 1.0
        combined_tensor.F[overlapping_mask_comb, 1:] = gt.features_at_coordinates(combined_tensor.C[overlapping_mask_comb].float())

        gt_interpolated = self.conv_sum(combined_tensor)

        # Interpolate raw features from the ground truth at the non-overlapping predicted locations
        raw_features = gt_interpolated.features_at_coordinates(prediction.C[~overlapping_mask].float())[:, 1:] / \
                       gt_interpolated.features_at_coordinates(prediction.C[~overlapping_mask].float())[:, 0].unsqueeze(1)

        # Create a tensor for ground truth on predicted coordinates
        gt_on_pred = ME.SparseTensor(
            coordinates=prediction.C,
            features=torch.zeros((prediction.F.shape[0], N-1), device=gt.C.device),
            device=prediction.C.device
        )

        # Set features for overlapping points and interpolate for non-overlapping points
        #overlapping_mask2 = utils.overlapping_mask(gt_on_pred, gt)
        gt_on_pred.F[overlapping_mask] = gt.features_at_coordinates(gt_on_pred.C[overlapping_mask].float())
        gt_on_pred.F[~overlapping_mask] = raw_features

        return gt_on_pred