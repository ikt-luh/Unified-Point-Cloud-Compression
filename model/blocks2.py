import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import copy

import utils
from compressai.layers import GDN


class CFE(torch.nn.Module):
    def __init__(self, N, N_q, encode=True):
        super().__init__()
        self.encode = encode

        self.conv_1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N, out_channels=N, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N, out_channels=N, kernel_size=3, stride=1, bias=True, dimension=3),
        )
        self.conv_2 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N, out_channels=N, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N, out_channels=N, kernel_size=3, stride=1, bias=True, dimension=3),
        )
        self.conv_3 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N, out_channels=N, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False)
        )
        self.conv_Q = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N_q, out_channels=N//2, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N//2, out_channels=N, kernel_size=1, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N, out_channels=N*2, kernel_size=1, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N*2, out_channels=N*2, kernel_size=1, stride=1, bias=True, dimension=3),
        )

    def forward(self, x, condition):
        x_res = ME.SparseTensor(coordinates=x.C,
                                features=x.F.clone(),
                                device=x.device,
                                tensor_stride=x.tensor_stride)
        
        x = self.conv_1(x)

        # Scale and shift
        condition = self.conv_Q(condition)
        beta, gamma = condition.features_at_coordinates(x.C.float()).chunk(2, dim=1)
        if self.encode:
            #feats = x.F * F.sigmoid(beta) + gamma
            feats = x.F * beta + gamma
        else:
            #feats = x.F / F.sigmoid(beta) - gamma
            feats = x.F * beta + gamma
         
        x = ME.SparseTensor(coordinates=x.C, 
                            features=feats, 
                            device=x.device, 
                            tensor_stride=x.tensor_stride)

        x = self.conv_2(x)

        x = ME.SparseTensor(coordinates=x.C,
                                features=x.F + x_res.features_at_coordinates(x.C.float()),
                                device=x.device,
                                tensor_stride=x.tensor_stride)

        #x = self.conv_3(x)
        return x


class GenerativeUpBlock(torch.nn.Module):
    def __init__(self, N_in, N_out):
        super().__init__()
        self.prune = ME.MinkowskiPruning()

        self.up_conv = ME.MinkowskiGenerativeConvolutionTranspose(in_channels=N_in, out_channels=N_out, kernel_size=3, stride=2, bias=True, dimension=3)

        self.conv_1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N_out, out_channels=N_out, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N_out, out_channels=N_out, kernel_size=3, stride=1, bias=True, dimension=3)
        )

        self.occ_predict = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N_out, out_channels=N_out, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N_out, out_channels=1, kernel_size=1, stride=1, bias=True, dimension=3),
        )


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

    def forward(self, x, k=None):
        x = self.up_conv(x)
        #x = self.conv_1(x)

        predictions = self.occ_predict(x)

        occupancy_mask = self._topk_prediction(predictions, k)
        up_coords = predictions.C[occupancy_mask]

        #x = self._prune_coords(x, up_coords) DO I NEED THIS?
        x = self.prune(x, occupancy_mask)
        return x, predictions, up_coords


class Pruning(nn.Module):
    def __init__(self):
        super().__init__()
        self.prune = ME.MinkowskiPruning()

    
    def forward(self, x, coords):
        x = self._prune_coords(x, coords)
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


class MinkowskiGDN(GDN):
    def __init__(self,
        in_channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
        kernel_size: int = 1,
    ):
        super(MinkowskiGDN, self).__init__(in_channels, inverse, beta_min, gamma_init)
        self.kernel_size = kernel_size


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, C = x.F.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)

        gamma = gamma.reshape(C, C, 1)

        feats = x.F.T.unsqueeze(0)
        norm = F.conv1d(torch.abs(feats), gamma, beta)
        norm = norm[0].T

        if not self.inverse:
            norm = 1.0 / norm

        output_features = x.F[:] * norm 
        return ME.SparseTensor(coordinates=x.C, features=output_features, tensor_stride=x.tensor_stride, device=x.device)

class MinkowskiGDNScaled(GDN):
    def __init__(self,
        in_channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
        kernel_size: int = 1,
    ):
        super(MinkowskiGDNScaled, self).__init__(in_channels, inverse, beta_min, gamma_init)

        self.scaling_nn = nn.Sequential(
            nn.Linear(2, in_channels//4),
            nn.ReLU(inplace=False),
            nn.Linear(in_channels//4, in_channels),
            nn.ReLU(inplace=False),
            nn.Linear(in_channels, in_channels),
            nn.Softplus(),
        )


    def forward(self, x: torch.Tensor, q) -> torch.Tensor:
        _, C = x.F.size()

        scale = self.scaling_nn(q)
            
        batch_indices = x.C[:, 0]
        scale_expanded = scale[0, batch_indices]


        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)

        gamma = gamma.reshape(C, C, 1)


        if not self.inverse:
            feats = (x.F * scale_expanded).T.unsqueeze(0)
            norm = F.conv1d(torch.abs(feats), gamma, beta)
            norm = norm[0].T
            norm = 1.0 / norm
            output_features = x.F[:] * norm 
        else:
            feats = x.F.T.unsqueeze(0)
            norm = F.conv1d(torch.abs(feats), gamma, beta)
            norm = norm[0].T
            output_features = (x.F[:] * norm) / scale_expanded

        #print(torch.min(output_features), torch.max(output_features))
        return ME.SparseTensor(coordinates=x.C, features=output_features, tensor_stride=x.tensor_stride, device=x.device)

"""

        self.conv = ME.MinkowskiConvolution(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=1, bias=True, dimension=3)

        # Define gamma and beta as Parameters with correct shapes
        gamma = self.conv.kernel
        beta = self.conv.bias

        # Initialize gamma and beta values
        with torch.no_grad():
            gamma.fill_(gamma_init)

        gamma = self.gamma_reparam.init(gamma)
        beta = self.beta_reparam.init(beta)
        self.gamma = nn.Parameter(gamma)
        self.beta = nn.Parameter(beta)

        self.scaling_nn = nn.Sequential(
            nn.Linear(2, in_channels//4),
            nn.ReLU(inplace=False),
            nn.Linear(in_channels//4, in_channels),
            nn.ReLU(inplace=False),
            nn.Linear(in_channels, in_channels),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor, q) -> torch.Tensor:
        print(self.parameters())
        _, C = x.F.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)

        scale = self.scaling_nn(q)
            
        batch_indices = x.C[:, 0]
        scale_expanded = scale[0, batch_indices]
        print(scale_expanded.shape)

        scaled_features = x.F * scale_expanded
        x = ME.SparseTensor(coordinates=x.C, features=scaled_features, tensor_stride=x.tensor_stride, device=x.device)

        self.conv.bias = nn.Parameter(beta)
        self.conv.kernel = nn.Parameter(gamma)
        print(self.gamma[0])
        norm = self.conv(x)

        if not self.inverse:
            output_features = x.F / norm.F 
        else:
            output_features = x.F * norm.F

        return ME.SparseTensor(coordinates=x.C, features=output_features, tensor_stride=x.tensor_stride, device=x.device)
"""