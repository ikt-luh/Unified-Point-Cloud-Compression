import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from model.blocks import *  


class AnalysisTransform(nn.Module):
    """
    Simple Analysis Module consisting of 3 blocks on Non-linear transformations
    """
    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Dictionary containing information for the transformation
            keys:
                C_in: Number of input channels
                N1: Filters in the 1st layer
                N2: Filters in the 2nd layer
                N3: Filters in the 3rd layer
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
        batch_indices = torch.unique(x.C[:, 0])  # Get unique batch IDs
        k_per_batch = []
        for batch_idx in batch_indices:
            k = (x.C[:, 0] == batch_idx).sum().item()
            k_per_batch.append(k)
        return k_per_batch
        
        

    def forward(self, x, q):
        """
        Forward pass for the analysis transform

        Parameters
        ----------
        x: ME.SparseTensor
            Sparse Tensor containing the orignal features

        returns
        -------
        x: ME.SparseTensor
            Sparse Tensor containing the latent features
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
        return x, q, k



class SparseSynthesisTransform(torch.nn.Module):
    """
    Sparse Decoder/ Synthesis Transform module for Attribute Compression
    Operates by pruning voxels after each upsampling step using the original point cloud geometry.
    """
    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Dictionary containing information for the transformation
            keys:
                C_in: Number of input channels
                N1: Filters in the 3rd layer
                N2: Filters in the 2nd layer
                N3: Filters in the 1st layer
        """
        super().__init__()

        C_out = config["C_out"]
        N1 = config["N1"]
        N2 = config["N2"]
        N3 = config["N3"]
        N4 = config["N4"]
 
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
        self.color_conv = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N1//4, out_channels=C_out, kernel_size=1, stride=1, bias=True, dimension=3),
        )

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


        self.prune = Pruning()
        
        # Auxiliary
        self.down_conv = ME.MinkowskiConvolution(in_channels=1, out_channels=1, kernel_size=3, stride=2, dimension=3)



    def forward(self, x, q_value, coords=None, k=None):
        """
        Forward pass for the synthesis transform

        Parameters
        ----------
        x: ME.SparseTensor
            Sparse Tensor containing the latent features
        coords: ME.SparseTensor
            Sparse Tensor containing coordinates of the upsampled point cloud

        returns
        -------
        x: ME.SparseTensor
            Sparse Tensor containing the upsampled features at location of coords
        """
        x = self.up_1(x)
        predict_1 = self.predict_1(x)
        occupancy_mask = self._topk_prediction(predict_1, k[0])
        coords_1 = x.C[occupancy_mask]
        x = self.prune(x, coords_1)

        x = self.up_2(x)
        predict_2 = self.predict_2(x)
        occupancy_mask = self._topk_prediction(predict_2, k[1])
        coords_2 = x.C[occupancy_mask]
        x = self.prune(x, coords_2)

        x = self.up_3(x)
        predict_3 = self.predict_3(x)
        occupancy_mask = self._topk_prediction(predict_3, k[2])
        coords_3 = x.C[occupancy_mask]
        x = self.prune(x, coords_3)

        x = self.color_conv(x)

        if coords is not None:
            predictions = [predict_1, predict_2, predict_3]
            
            # Generate ground truth values
            with torch.no_grad():
                points_1 = self.down_conv(coords)
                points_2 = self.down_conv(points_1)
            points = [points_2, points_1, coords]
            return x, points, predictions
        else:
            return x


    def _topk_prediction(self, prediction, k):
        """
        Mask the top-k elements for each batch in prediction to get attributes at predicted points.
        """
        batch_indices = torch.unique(prediction.C[:, 0])  # Get unique batch IDs
        pred_occupancy_mask = torch.zeros_like(prediction.F[:, 0], dtype=torch.bool)

        for batch_idx in batch_indices:
            # Mask for current batch
            current_batch_mask = prediction.C[:, 0] == batch_idx

            # Extract the predictions for the current batch and get top-k
            current_preds = prediction.F[current_batch_mask, 0]
            current_k = k[batch_idx]
            _, top_indices = torch.topk(current_preds, int(current_k))
    
            # Use advanced indexing to set the top-k indices to True
            indices_for_current_batch = torch.nonzero(current_batch_mask).squeeze()
            pred_occupancy_mask[indices_for_current_batch[top_indices]] = True

        return pred_occupancy_mask


    def _prune_coords(self, x, occupied_points=None):
        """
        Prunes the coordinates after upsampling, only keeping points coinciding with occupied points

        Parameters
        ----------
        x: ME.SparseTensor
            Upsampled point cloud with features
        occupied_points: ME.SparseTensor
            Sparse Tensor containing the coordinates to keep

        returns
        -------
        x: ME.SparseTensor
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