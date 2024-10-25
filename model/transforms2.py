import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from model.blocks2 import *  


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
        N2_q = config["N2_q"]
        N3_q = config["N3_q"]
        N4_q = config["N4_q"]

        self.cond_conv = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N1, out_channels=N1//2, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N1//2, out_channels=2, kernel_size=3, stride=1, bias=True, dimension=3),
        )

        self.pre_conv = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=C_in, out_channels=N1, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
        )

        self.down_1 = ME.MinkowskiConvolution(in_channels=N1, out_channels=N2, kernel_size=3, stride=2, bias=True, dimension=3)
        self.down_2 = ME.MinkowskiConvolution(in_channels=N2, out_channels=N3, kernel_size=3, stride=2, bias=True, dimension=3)
        self.down_3 = ME.MinkowskiConvolution(in_channels=N3, out_channels=N4, kernel_size=3, stride=2, bias=True, dimension=3)

        self.scale_1 = CFE(N2, N2_q, encode=True)
        self.scale_2 = CFE(N3, N3_q, encode=True)
        self.scale_3 = CFE(N4, N4_q, encode=True)

        self.post_conv = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N4 + N4_q, out_channels=N4 + N4_q, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N4 + N4_q, out_channels=N4, kernel_size=3, stride=1, bias=True, dimension=3),
        )

        # Q map
        self.Q_down_1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=2, out_channels=N2_q, kernel_size=3, stride=2, bias=True, dimension=3),
            ME.MinkowskiConvolution(in_channels=N2_q, out_channels=N2_q, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N2_q, out_channels=N2_q, kernel_size=3, stride=1, bias=True, dimension=3),
        )
        self.Q_down_2 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N2_q, out_channels=N3_q, kernel_size=3, stride=2, bias=True, dimension=3),
            ME.MinkowskiConvolution(in_channels=N3_q, out_channels=N3_q, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N3_q, out_channels=N3_q, kernel_size=3, stride=1, bias=True, dimension=3),
        )
        self.Q_down_3 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N3_q, out_channels=N4_q, kernel_size=3, stride=2, bias=True, dimension=3),
            ME.MinkowskiConvolution(in_channels=N4_q, out_channels=N4_q, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N4_q, out_channels=N4_q, kernel_size=3, stride=1, bias=True, dimension=3),
        )
        self.Q_post_conv = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N4_q, out_channels=N4_q, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N4_q, out_channels=N4_q, kernel_size=3, stride=1, bias=True, dimension=3),
        )


    def count_per_batch(self, x):
        batch_indices = torch.unique(x.C[:, 0])  # Get unique batch IDs
        k_per_batch = []
        for batch_idx in batch_indices:
            k = (x.C[:, 0] == batch_idx).sum().item()
            k_per_batch.append(k)
        return k_per_batch
        
        

    def forward(self, x, Q):
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

        # Pre-Conv
        x = self.pre_conv(x)
        Q_plus = self.cond_conv(x)
        Q = ME.SparseTensor(
            coordinates=Q.C,
            features=Q.F * F.sigmoid(Q_plus.features_at_coordinates(Q.C.float())),
            device=Q.device
        )

        # Level 1
        x = self.down_1(x)
        Q = self.Q_down_1(Q)
        x = self.scale_1(x, Q)
        k.append(self.count_per_batch(x))

        # Layer 2
        x = self.down_2(x)
        Q = self.Q_down_2(Q)
        x = self.scale_2(x, Q)
        k.append(self.count_per_batch(x))

        # Layer 3
        x = self.down_3(x)
        Q = self.Q_down_3(Q)
        x = self.scale_3(x, Q)

        # Concat quality and features for compression
        Q = self.Q_post_conv(Q)
        x = ME.SparseTensor(coordinates=x.C,
                            features=torch.cat([x.F, Q.features_at_coordinates(x.C.float())], dim=1),
                            tensor_stride=x.tensor_stride)

        x = self.post_conv(x)

        k.reverse()
        return x, Q, k


        


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
        self.N4 = config["N4"]
        N2_q = config["N2_q"]
        N3_q = config["N3_q"]
        N4_q = config["N4_q"]
        
        self.pre_conv = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=self.N4, out_channels=self.N4 + N4_q, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=self.N4 + N4_q, out_channels=self.N4 + N4_q, kernel_size=3, stride=1, bias=True, dimension=3)
        )

        self.Q_pre_conv = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N4_q, out_channels=N4_q, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N4_q, out_channels=N4_q, kernel_size=3, stride=1, bias=True, dimension=3),
        )        
        self.Q_conv_1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N4_q, out_channels=N4_q, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N4_q, out_channels=N4_q, kernel_size=3, stride=1, bias=True, dimension=3),
        )
        self.Q_conv_2 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N3_q, out_channels=N3_q, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N3_q, out_channels=N3_q, kernel_size=3, stride=1, bias=True, dimension=3),
        )
        self.Q_conv_3 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N2_q, out_channels=N2_q, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N2_q, out_channels=N2_q, kernel_size=3, stride=1, bias=True, dimension=3),
        )
        self.Q_up_1 = ME.MinkowskiGenerativeConvolutionTranspose(in_channels=N4_q, out_channels=N3_q, kernel_size=3, stride=2, bias=True, dimension=3)
        self.Q_up_2 = ME.MinkowskiGenerativeConvolutionTranspose(in_channels=N3_q, out_channels=N2_q, kernel_size=3, stride=2, bias=True, dimension=3)


        # Model
        self.up_1 = GenerativeUpBlock(self.N4, N3)
        self.up_2 = GenerativeUpBlock(N3, N2)
        self.up_3 = GenerativeUpBlock(N2, N1)

        self.scale_1 = CFE(self.N4, N4_q, encode=False)
        self.scale_2 = CFE(N3, N3_q, encode=False)
        self.scale_3 = CFE(N2, N2_q, encode=False)

        self.post_conv = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N1, out_channels=N1, kernel_size=1, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N1, out_channels=N1, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N1, out_channels=C_out, kernel_size=1, stride=1, bias=True, dimension=3),
        )

        self.prune = Pruning()
        
        # Auxiliary
        self.down_conv = ME.MinkowskiConvolution(in_channels=1, out_channels=1, kernel_size=3, stride=2, dimension=3)



    def forward(self, x, Q, coords=None, k=None):
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
        x_full = self.pre_conv(x)
        x = ME.SparseTensor(
            coordinates=x_full.C,
            features=x_full.F[:, :self.N4],
            tensor_stride=x_full.tensor_stride,
            device=x_full.device
        )
        Q = ME.SparseTensor(
            coordinates=x_full.C,
            features=x_full.F[:, self.N4:],
            tensor_stride=x_full.tensor_stride,
            device=x_full.device
        )

        # Level 3
        Q = self.Q_conv_1(Q)
        x = self.scale_1(x, Q)

        x, predict_2, up_coords = self.up_1(x, k=k[0])
        Q = self.Q_up_1(Q)
        Q = self.prune(Q, up_coords)

        # Layer 2
        Q = self.Q_conv_2(Q)
        x = self.scale_2(x, Q)

        x, predict_1, up_coords = self.up_2(x, k=k[1])
        Q = self.Q_up_2(Q, up_coords)
        Q = self.prune(Q, up_coords)

        # Level 1
        Q = self.Q_conv_3(Q)
        x = self.scale_3(x, Q)

        x, predict_final, up_coords = self.up_3(x, k=k[2])

        # Post Conv
        x = self.post_conv(x)

        if coords is not None:
            predictions = [predict_2, predict_1, predict_final]
            with torch.no_grad():
                points_1 = self.down_conv(coords)
                points_2 = self.down_conv(points_1)
            points = [points_2, points_1, coords]
            return x, points, predictions
        else:
            return x

