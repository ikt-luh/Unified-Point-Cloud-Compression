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
        Parameters
        ----------
        config: dict
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
                case "ColorSSIM":
                    self.losses[id] = ColorSSIM(setting)
                case "FocalLoss":
                    self.losses[id] = FocalLoss(setting)
                case "Multiscale_FocalLoss":
                    self.losses[id] = Multiscale_FocalLoss(setting)
                case _:
                    print("Not found {}".format(key))
                
                    
    def __call__(self, gt, pred):
        """
        Call the loss function to return sum of all losses
        
        Parameters
        ----------
        gt: ME.SparseTensor
            Ground truth point cloud
        pred: dict
            Dictionary containing information for computing the loss

        returns
        -------
        total_loss: torch.tensor
            Total loss after adding and weighting
        losses: dict
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

        # Identify non-overlapping coordinates
        overlapping_mask = torch.isin(gt_flat, pred_flat)

        pred_colors = prediction.features_at_coordinates(gt.C[overlapping_mask].float())
        gt_colors = gt.F[overlapping_mask]

        color_loss = self.loss_func(gt_colors, pred_colors) 
        color_loss *= q_map.features_at_coordinates(gt.C[overlapping_mask].float())[:, 1].unsqueeze(1)

        return color_loss.mean()
    

class FocalLoss():
    """
    Focal Loss for predicted voxels
    for gamma = 0, this is BCE
    """
    def __init__(self, config):
        self.identifier = config["id"]
        self.alpha = config["alpha"]
        self.gamma = config["gamma"]

    def __call__(self, gt, pred):
        prediction = pred["prediction"]

        # Convert 3D coordinates to a flattened representation
        scaling_factors = torch.tensor([1, 1e4, 1e8, 1e12], dtype=torch.int64, device=gt.C.device)
        gt_flat = (gt.C.to(torch.int64) * scaling_factors).sum(dim=1)
        pred_flat = (prediction.C.to(torch.int64) * scaling_factors).sum(dim=1)

        # Identify non-overlapping coordinates
        overlapping_mask = torch.isin(pred_flat, gt_flat)
        
        # Classificaton
        p_z = F.sigmoid(prediction.F[:, 0]+0.5) # F[0] contains occupancy
        
        # Build pt_z and alpha_z
        pt_z = torch.where(overlapping_mask, p_z, 1 - p_z)
        alpha_z = torch.where(overlapping_mask, self.alpha, 1 - self.alpha)
        pt_z = torch.clip(pt_z, 1e-2, 1)

        # Focal Loss
        focal_loss = - alpha_z * (1-pt_z)**self.gamma * torch.log(pt_z)
        focal_loss = focal_loss.mean()

        return focal_loss * pred["lambdas"][0][0]

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
            # Convert 3D coordinates to a flattened representation
            scaling_factors = torch.tensor([1, 1e4, 1e9, 1e14], dtype=torch.int64, device=gt.C.device)
            gt_flat = (coords.C.to(torch.int64) * scaling_factors).sum(dim=1)
            pred_flat = (prediction.C.to(torch.int64) * scaling_factors).sum(dim=1)

            # Identify non-overlapping coordinates
            overlapping_mask = torch.isin(pred_flat, gt_flat)
        
            # Classificaton
            p_z = F.sigmoid(prediction.F[:, 0]) # F[0] contains occupancy
        
            # Build pt_z and alpha_z
            pt_z = torch.where(overlapping_mask, p_z, 1 - p_z)
            alpha_z = torch.where(overlapping_mask, self.alpha, 1 - self.alpha)
            pt_z = torch.clip(pt_z, 1e-2, 1)

            # Focal Loss
            focal_loss = - alpha_z * (1-pt_z)**self.gamma * torch.log(pt_z)
            
            # Q Map
            q_avgs = self.pooling(q_map, coordinates=prediction.C)
            q_map = self.down_pooling(q_map)

            loss += (focal_loss * q_avgs.features_at_coordinates(prediction.C.float())[:, 0]).mean()

        return loss



class ShepardsLoss():
    """
    Shepard's Loss, using L2/L1 color loss on ground truth voxel locations.
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

        # Create 3D window for convolution
        self.window = self.create_window_3D(self.window_size)

        # Define Minkowski convolutions
        self.conv_sum = self._init_minkowski_conv(4, self.window_size, self.window)
        #self.conv_sum_q = self._init_minkowski_conv(3, self.window_size, self.window)

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

        Parameters
        ----------
        window_size: int
            Size of the window (must be odd for a symmetric ball)

        Returns
        ----------
        window: torch.tensor
            3D window of shape (window_size**3) with inverse distance weighting
        """
        radius = window_size // 2
        window = torch.zeros((window_size, window_size, window_size))

        # Create 3D grid
        z, y, x = torch.meshgrid(
            torch.arange(window_size) - radius,
            torch.arange(window_size) - radius,
            torch.arange(window_size) - radius
        )

        distance = torch.sqrt(x**2 + y**2 + z**2)
        window = 1 / (distance ** self.p + 1e-5)
        window[distance > radius] = 0

        #window = window / window.sum()
        window = window.view(-1, 1)
        return window

    def __call__(self, gt, pred):
        """
        Forward pass for Shepard's loss calculation.
        """
        self.conv_sum.to(gt.C.device)
        #self.conv_sum_q.to(gt.C.device)

        prediction = pred["prediction"]
        q_map = pred["q_map"]

        gt_on_pred = self.interpolate_gt_to_pred(gt, prediction)
        #q_map_on_pred = self.interpolate_gt_to_pred(q_map, prediction, interpolate_q_map=True)

        valid_mask = (~torch.isnan(gt_on_pred.F) & ~torch.isinf(gt_on_pred.F)).all(dim=1)
        batch_indicies = gt_on_pred.C[valid_mask, 0]
        color_loss = self.loss_func(gt_on_pred.F[valid_mask], prediction.F[valid_mask]) * q_map.F[batch_indicies, 1].unsqueeze(1)
        
        return color_loss.mean()


    def interpolate_gt_to_pred(self, gt, prediction, interpolate_q_map=False):
        """
        Interpolate ground truth values to predicted coordinates using Minkowski convolution.
        """      
        N = 3 if interpolate_q_map else 4

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

        if interpolate_q_map:
            gt_interpolated = self.conv_sum_q(combined_tensor)
        else:
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
        overlapping_mask2 = utils.overlapping_mask(gt_on_pred, gt)
        gt_on_pred.F[overlapping_mask2] = gt.features_at_coordinates(gt_on_pred.C[overlapping_mask2].float())
        gt_on_pred.F[~overlapping_mask2] = raw_features

        return gt_on_pred






class ColorSSIM():
    def __init__(self, config):
        self.identifier = config["id"]
        self.window_size = config["window_size"]
        self.yuv = config["yuv"]
        self.window = self.create_window_3D(self.window_size)

        self.conv_sum = ME.MinkowskiChannelwiseConvolution(in_channels=30, kernel_size=self.window_size, stride=1, dimension=3)
        self.conv_sum.kernel = torch.nn.Parameter(self.window)
        self.conv_sum.kernel.requires_grad = False

        self.C1 = (0.01)**2
        self.C2 = (0.03)**2
        self.C3 = self.C2 / 2


    def gaussian(self, window_size, sigma):
        """
        Computes a 1D Gaussian window

        Parameters
        ----------
        window_size: int
            Size of the window
        sigma: float
            Sigma for computing the gaussian kernel

        returns
        ----------
        gauss: torch.tensor
            Normalized gaussian window of length window_size
        """
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()



    def create_window_3D(self, window_size):
        """
        Compute a 3D Gaussian kernel of defined window_size

        Parameters
        ----------
        window_size: int
            Size of the window

        returns
        ----------
        window: torch.tensor
            3D Gaussian window of shape window_size**3
        """
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t())
        _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
        window = _3D_window.view(-1, 1)
        #window = torch.ones_like(window)
        return window

    def rgb_to_yuv(self, rgb_tensor):
        """
        Convert a rgb_tensor to RGB (move to utils?)
        BT.709

        Parameters
        ----------
        rgb_tensor: torch.tensor
            RGB tensor of shape Nx3
        
        returns
        ----------
        yuv_reshaped: torch.tensor
            YUV tensor of shape Nx3
        """
        color_matrix = torch.tensor([
                                    [0.2126, 0.7152, 0.00722],
                                    [-0.1146, -0.3854, 0.5],
                                    [0.5, -0.4542, 0.0458]
                                ])
        color_matrix = color_matrix.to(rgb_tensor.device)

        # Matrix multiplication
        yuv_reshaped = torch.einsum('ij,nj->ni', color_matrix, rgb_tensor)
        yuv_reshaped[:, 1] += 0.5
        yuv_reshaped[:, 2] += 0.5

        # Reshape back to original size
        return yuv_reshaped

    def __call__(self, gt, pred):
        device = gt.device
        self.conv_sum.to(device)

        prediction = pred["prediction"]
        q_map = pred["q_map"]

        #predicted_colors = self.mask_topk_prediction(gt, prediction)
        predicted_colors = ME.SparseTensor(coordinates=prediction.C, features=prediction.F)
        #predicted_colors = ME.SparseTensor(coordinates=predicted_colors.C, features=predicted_colors.F)
        if self.yuv:
            gt = ME.SparseTensor(coordinates=gt.C, features=self.rgb_to_yuv(gt.F))
            predicted_colors = ME.SparseTensor(coordinates=predicted_colors.C, features=self.rgb_to_yuv(predicted_colors.F))

        # Convert 3D coordinates to a flattened representation
        scaling_factors = torch.tensor([1, 1e4, 1e9, 1e14], dtype=torch.int64, device=gt.C.device)
        gt_flat = (gt.C.to(torch.int64) * scaling_factors).sum(dim=1)
        pred_flat = (prediction.C.to(torch.int64) * scaling_factors).sum(dim=1)

        # Identify non-overlapping coordinates
        #overlapping_mask = torch.isin(pred_flat, gt_flat)
        #union_coordinates = prediction.C[overlapping_mask]

        union_coordinates = torch.unique(torch.cat([gt.C, prediction.C], 0), dim=0)

        result = self.convolutions(gt, predicted_colors, union_coordinates)

        # Correction factors
        N_x_inv = torch.where(result["N_x"] > 0.0, 1 / result["N_x"], 0)# 1e-10)
        N_y_inv = torch.where(result["N_y"] > 0.0, 1 / result["N_y"], 0)# 1e-10)
        N_xy_inv = torch.where(result["N_xy"] > 0.0, 1 / result["N_xy"],0)# 1e-10)

        #N_x_inv_corr = torch.where(result["N_x"] > 1.0, 1 / (result["N_x"] - 1), 0.0)
        #N_y_inv_corr = torch.where(result["N_y"] > 1.0, 1 / (result["N_y"] - 1), 0.0)
        #N_xy_inv_corr = torch.where(result["N_xy"] > 1.0, 1 / (result["N_xy"] - 1), 0.0)

        # Compute means
        mu_x = N_x_inv * result["sum_x"]
        mu_y = N_y_inv * result["sum_y"]
        mu_x_masked = N_xy_inv * result["m_sum_x"]
        mu_y_masked = N_xy_inv * result["m_sum_y"]

        # Compute variances
        #sigma_x_sq = N_x_inv_corr * (result["sum_x_sq"] - result["N_x"] * mu_x.pow(2))
        #sigma_y_sq = N_y_inv_corr * (result["sum_y_sq"] - result["N_y"] * mu_y.pow(2))
        #sigma_x_sq_masked = N_xy_inv_corr * (result["m_sum_x_sq"] - result["N_xy"] * mu_x_masked.pow(2))
        #sigma_y_sq_masked = N_xy_inv_corr * (result["m_sum_y_sq"] - result["N_xy"] * mu_y_masked.pow(2))
        sigma_x_sq = N_x_inv * result["sum_x_sq"] - mu_x.pow(2)
        sigma_y_sq = N_y_inv * result["sum_y_sq"] - mu_y.pow(2)
        sigma_x_sq_masked = N_xy_inv * result["m_sum_x_sq"] - mu_x_masked.pow(2)
        sigma_y_sq_masked = N_xy_inv * result["m_sum_y_sq"] - mu_y_masked.pow(2)

        # Compute stddev
        sigma_x_sq = torch.where(sigma_x_sq > 0.0, sigma_x_sq, 0) #1e-10)
        sigma_y_sq = torch.where(sigma_y_sq > 0.0, sigma_y_sq, 0) #1e-10)
        sigma_x_sq_masked = torch.where(sigma_x_sq_masked > 0.0, sigma_x_sq_masked, 0) #1e-10)
        sigma_y_sq_masked = torch.where(sigma_y_sq_masked > 0.0, sigma_y_sq_masked, 0) #1e-10)

        sigma_x = torch.sqrt(sigma_x_sq)
        sigma_y = torch.sqrt(sigma_y_sq)
        sigma_x_masked = torch.sqrt(sigma_x_sq_masked)
        sigma_y_masked = torch.sqrt(sigma_y_sq_masked)

        # Compute covariance
        #sigma_xy = N_xy_inv_corr * (result["m_sum_xy"] -  result["N_xy"] * mu_x_masked * mu_y_masked)
        sigma_xy = N_xy_inv * result["m_sum_xy"] -  mu_x_masked * mu_y_masked

        # Factors
        luminance = (2 * mu_x * mu_y + self.C1)  / (mu_x.pow(2) + mu_y.pow(2) + self.C1)
        lightness = (2 * sigma_x * sigma_y + self.C2) / (sigma_x_sq + sigma_y_sq + self.C2)
        structure = (sigma_xy + self.C3)  / (sigma_x_masked * sigma_y_masked + self.C3)

        # SSIM
        ssim = luminance * structure * lightness
        ssim = (((1 - ssim) / 2)) * q_map.features_at_coordinates(union_coordinates.float())[:, 1].unsqueeze(1)
        if self.yuv:
            ssim *= torch.tensor([[0.75, 0.125, 0.125]], device=ssim.device)

        return ssim.mean() 

    def mask_topk_prediction(self, gt, prediction):
        """
        Mask the top-k elements for each batch in prediction to get attributes at predicted points.
        """
        batch_indices = torch.unique(prediction.C[:, 0])  # Get unique batch IDs
        pred_occupancy_mask = torch.zeros_like(prediction.F[:, 0], dtype=torch.bool)

        for batch_idx in batch_indices:
            # Mask for current batch
            current_batch_mask = prediction.C[:, 0] == batch_idx
            # Get the k value for the current batch from gt
            k = (gt.C[:, 0] == batch_idx).sum().item()

            # Extract the predictions for the current batch and get top-k
            current_preds = prediction.F[current_batch_mask, 0]
            _, top_indices = torch.topk(current_preds, k)
    
            # Use advanced indexing to set the top-k indices to True
            indices_for_current_batch = torch.nonzero(current_batch_mask).squeeze()
            pred_occupancy_mask[indices_for_current_batch[top_indices]] = True

        predicted_colors = ME.SparseTensor(coordinates=prediction.C[pred_occupancy_mask], features=prediction.F[pred_occupancy_mask])

        return predicted_colors


    def convolutions(self, gt, prediction, union_coordinates):
        """
        Perform convolutions on all feature maps at once for computational efficiency. 
        """
        # Compute occupancies
        gt_occupancy = ME.SparseTensor(coordinates=gt.C, features=torch.ones(gt.C.shape[0], device=gt.F.device).unsqueeze(1))
        pred_occupancy = ME.SparseTensor(coordinates=prediction.C, features=torch.ones(prediction.C.shape[0], device=gt.F.device).unsqueeze(1))
        #gt_occupancy = ME.SparseTensor(coordinates=union_coordinates, features=torch.ones(union_coordinates.shape[0], device=gt.F.device).unsqueeze(1))
        #pred_occupancy = ME.SparseTensor(coordinates=union_coordinates, features=torch.ones(union_coordinates.shape[0], device=gt.F.device).unsqueeze(1))

        gt_occupancy = ME.SparseTensor(coordinates=union_coordinates, features=gt_occupancy.features_at_coordinates(union_coordinates.float()))
        pred_occupancy = ME.SparseTensor(coordinates=union_coordinates, features=pred_occupancy.features_at_coordinates(union_coordinates.float()))
        shared_occupancy = ME.SparseTensor(coordinates=union_coordinates, features=pred_occupancy.F * gt_occupancy.F)

        # Colors on union coordinates
        predicted_colors_union = ME.SparseTensor(coordinates=union_coordinates, features=prediction.features_at_coordinates(union_coordinates.float()) * pred_occupancy.F)
        gt_colors_union = ME.SparseTensor(coordinates=union_coordinates, features=gt.features_at_coordinates(union_coordinates.float()) * gt_occupancy.F) 
        #pred_gt_colors = ME.SparseTensor(coordinates=union_coordinates, features=predicted_colors_union.F * gt_colors_union.features_at_coordinates(union_coordinates.float()))

        # Masked colors on intersection coordinates
        predicted_colors_masked = ME.SparseTensor(coordinates=union_coordinates, features=predicted_colors_union.F * shared_occupancy.features_at_coordinates(union_coordinates.float()))
        gt_colors_masked = ME.SparseTensor(coordinates=union_coordinates, features=gt_colors_union.F * shared_occupancy.features_at_coordinates(union_coordinates.float()))
        pred_gt_colors_masked = ME.SparseTensor(coordinates=union_coordinates, features=predicted_colors_masked.F * gt_colors_masked.features_at_coordinates(union_coordinates.float()))

        # Do the convolution
        conv_tensor = ME.SparseTensor(
            coordinates=union_coordinates,
            features=torch.cat([
                gt_occupancy.F,     #0
                pred_occupancy.F,   #1
                shared_occupancy.F, #2

                gt_colors_union.F,  #3:5
                predicted_colors_union.F, #6:8
                gt_colors_union.F.pow(2), #9:11
                predicted_colors_union.F.pow(2), #12:14

                gt_colors_masked.F, #15:17
                predicted_colors_masked.F, #18:20
                gt_colors_masked.F.pow(2), #21:23
                predicted_colors_masked.F.pow(2), #24:26

                pred_gt_colors_masked.F, #27:29
            ], dim=1)
        )

        result = self.conv_sum(conv_tensor)

        result = {
            "N_x": result.F[:, 0].unsqueeze(1),
            "N_y": result.F[:, 1].unsqueeze(1),
            "N_xy": result.F[:, 2].unsqueeze(1),
            "sum_x": result.F[:, 3:6],
            "sum_y": result.F[:, 6:9],
            "sum_x_sq": result.F[:, 9:12],
            "sum_y_sq": result.F[:, 12:15],
            "m_sum_x": result.F[:, 15:18],
            "m_sum_y": result.F[:, 18:21],
            "m_sum_x_sq": result.F[:, 21:24],
            "m_sum_y_sq": result.F[:, 24:27],
            "m_sum_xy": result.F[:, 27:30],
        }
        return result