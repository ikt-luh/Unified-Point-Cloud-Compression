import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from model.blocks import *  


class AnalysisTransform(nn.Module):
    """
    Simple Analysis Module consisting of 3 blocks of non-linear transformations
    """
    def __init__(self, config):
        """
        Parameters:
            config (dict):
                Dictionary containing information for the transformation
                - C_in: Number of input channels
                - N1: Filters in the 1st level
                - N2: Filters in the 2nd level
                - N3: Filters in the 3rd level
                - N3: Filters in the bottleneck
        """
        super().__init__()

        C_in = config["C_in"]
        N1 = config["N1"]
        N2 = config["N2"]
        N3 = config["N3"]
        N4 = config["N4"]

        self.down_conv_1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=C_in, out_channels=N1, kernel_size=5, stride=2, bias=True, dimension=3),
            MinkowskiGDN(N1),
        )
        self.down_conv_2 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N1, out_channels=N2, kernel_size=5, stride=2, bias=True, dimension=3),
            MinkowskiGDN(N2),
        )
        self.down_conv_3 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N2, out_channels=N3, kernel_size=5, stride=2, bias=True, dimension=3),
            MinkowskiGDN(N3),
            ME.MinkowskiConvolution(in_channels=N3, out_channels=N4, kernel_size=5, stride=1, bias=True, dimension=3)
        )


    def count_per_batch(self, x):
        """
        Utility to count the number of points per batch

        Parameters:
            x (ME.SparseTensor):
                Sparse tensor for batch counting
        
        Returns:
            k_per_batch (list):
                List of the number of points per batch
        """
        batch_indices = torch.unique(x.C[:, 0])  
        k_per_batch = []
        for batch_idx in batch_indices:
            k = (x.C[:, 0] == batch_idx).sum().item()
            k_per_batch.append(k)
        return k_per_batch
        
        

    def forward(self, x):
        """
        Forward pass for the analysis transform.

        Parameters:
            x (ME.SparseTensor):
                Sparse Tensor of the point cloud

        Returns:
            y (ME.SparseTensor):
                Sparse Tensor containing the latent features
            k (list):
                List containing the number of points per batch at each level
        """
        k = []
        k.append(self.count_per_batch(x))

        # Level 1
        x = self.down_conv_1(x)
        k.append(self.count_per_batch(x))

        # Layer 2
        x = self.down_conv_2(x)
        k.append(self.count_per_batch(x))

        # Layer 3
        x = self.down_conv_3(x)

        k.reverse()
        return x, k



class SparseSynthesisTransform(torch.nn.Module):
    """
    Sparse Decoder/ Synthesis Transform module. 
    Operates by pruning voxels after each upsampling step using the original point cloud geometry.
    """
    def __init__(self, config):
        """
        Parameters:
            config (dict):
                Dictionary containing information for the transformation
                - C_in: Number of input channels
                - N1: Filters in the 1rd level
                - N2: Filters in the 2nd level
                - N3: Filters in the 3rd level
                - N4: Filters in the bottleneck
        """
        super().__init__()

        C_out = config["C_out"]
        N1 = config["N1"]
        N2 = config["N2"]
        N3 = config["N3"]
        N4 = config["N4"]
 
        # Up-Sampling Blocks
        self.up_1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N4, out_channels=N3, kernel_size=5, stride=1, bias=True, dimension=3),
            MinkowskiGDN(N3, inverse=True),
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=N3, out_channels=N2, kernel_size=5, stride=2, bias=True, dimension=3)
        )
        self.up_2 = nn.Sequential(
            MinkowskiGDN(N2, inverse=True),
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=N2, out_channels=N1, kernel_size=5, stride=2, bias=True, dimension=3)
        )
        self.up_3 = nn.Sequential(
            MinkowskiGDN(N1, inverse=True),
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels=N1, out_channels=N1//4, kernel_size=5, stride=2, bias=True, dimension=3)
        )

        # Final Color Convolution
        self.color_conv = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N1//4, out_channels=C_out, kernel_size=1, stride=1, bias=True, dimension=3),
        )

        # Occupancy prediction heads
        self.predict_1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N2, out_channels=N2//2, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N2//2, out_channels=1, kernel_size=3, stride=1, bias=True, dimension=3),
        )
        self.predict_2 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N1, out_channels=N1//2, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N1//2, out_channels=1, kernel_size=3, stride=1, bias=True, dimension=3),
        )
        self.predict_3 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N1//4, out_channels=N4//8, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N4//8, out_channels=1, kernel_size=3, stride=1, bias=True, dimension=3),
        )

        # Pruning
        self.prune = ME.MinkowskiPruning()
        
        # Auxiliary
        self.down_conv = ME.MinkowskiConvolution(in_channels=1, out_channels=1, kernel_size=3, stride=2, dimension=3)



    def forward(self, y, coords=None, k=None):
        """
        Forward pass for the synthesis transform.

        Parameters:
            y (ME.SparseTensor):
                Sparse Tensor containing the latent features
            coords (ME.SparseTensor):
                Sparse Tensor containing coordinates of the upsampled point cloud (optional, for training)
            k (list):
                List containing the number of points per level

        Returns:
            x (ME.SparseTensor):
                Sparse Tensor containing the upsampled features at location of coords
            points (list):
                Optional list of GT-Points for loss computation (training)
            predictions (list):
                Optional list of per-level occupancy prediction for loss computation (training)
        """
        # 3rd Level
        x = self.up_1(y)
        predict_1 = self.predict_1(x)
        occupancy_mask = self._topk_prediction(predict_1, k[0])
        coords_1 = x.C[occupancy_mask]
        x = self._prune_tensor(x, coords_1)

        # 2nd Level
        x = self.up_2(x)
        predict_2 = self.predict_2(x)
        occupancy_mask = self._topk_prediction(predict_2, k[1])
        coords_2 = x.C[occupancy_mask]
        x = self._prune_tensor(x, coords_2)

        # 1st Level
        x = self.up_3(x)
        predict_3 = self.predict_3(x)
        occupancy_mask = self._topk_prediction(predict_3, k[2])
        coords_3 = x.C[occupancy_mask]
        x = self._prune_tensor(x, coords_3)

        # Color pred. head
        x = self.color_conv(x)

        if coords is not None:
            # Training
            predictions = [predict_1, predict_2, predict_3]
            
            with torch.no_grad():
                points_1 = self.down_conv(coords)
                points_2 = self.down_conv(points_1)
            points = [points_2, points_1, coords]
            return x, points, predictions
        else:
            # Inference
            return x


    def _topk_prediction(self, prediction, k):
        """
        Generate a mask for the top-k elements for each batch in prediction to get attributes at predicted points.

        Parameters:
            prediction (ME.SparseTensor):
                Sparse Tensor containing the occupancy predictions
            k (list):
                List of top-k elements per batch
        
        Returns:
            pred_occupancy_mask (torch.tensor):
                Mask for the top-k predictions per batch
        """
        batch_indices = torch.unique(prediction.C[:, 0])  
        pred_occupancy_mask = torch.zeros_like(prediction.F[:, 0], dtype=torch.bool)
        for batch_idx in batch_indices:
            current_batch_mask = prediction.C[:, 0] == batch_idx

            current_preds = prediction.F[current_batch_mask, 0]
            current_k = k[batch_idx]
            _, top_indices = torch.topk(current_preds, int(current_k))
    
            indices_for_current_batch = torch.nonzero(current_batch_mask).squeeze()
            pred_occupancy_mask[indices_for_current_batch[top_indices]] = True

        return pred_occupancy_mask


    def _prune_tensor(self, x, occupied_points=None):
        """
        Prunes the coordinates after upsampling, only keeping points coinciding with occupied points

        Parameters:
            x (ME.SparseTensor):
                Upsampled point cloud with features
            occupied_points (ME.SparseTensor):
                Sparse Tensor containing the coordinates to keep

        Returns:
            x (ME.SparseTensor):
                Pruned tensor with features
        """
        # Define Scaling Factors
        scaling_factors = torch.tensor([1, 1e5, 1e10, 1e15], dtype=torch.int64, device=x.C.device)

        # Transform to unique indices
        x_flat = (x.C.to(torch.int64) * scaling_factors).sum(dim=1)
        guide_flat = (occupied_points.to(torch.int64) * scaling_factors).sum(dim=1)

        # Prune
        mask = torch.isin(x_flat, guide_flat)
        x = self.prune(x, mask)

        return x
