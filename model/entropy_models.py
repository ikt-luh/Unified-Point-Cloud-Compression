import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from compressai.ops import ops, LowerBound
from compressai.models.base import CompressionModel
from compressai.entropy_models import EntropyBottleneck, GaussianConditional #, EntropyBottleneckVbr
from utils import sort_points, sort_tensor

def quantize_noise(x):
    """
    Additive uniform noise quantization y = x + u with u being uniform noise from [-0.5, 0.5)
    
    Parameters: 
        x (torch.tensor):
            Input data tensor
    Returns:
        y (torch.tensor):
            Output tensor with per-element additive uniform noise
    """
    half = float(0.5)
    noise = torch.empty_like(x).uniform_(-half, half)
    y = x + noise
    return y


class SortedMinkowskiConvolution(ME.MinkowskiConvolution):
    """
    A Sorted MinkowskiConvolution to allow deterministic computation on the same hardware
    """
    def forward(self, input):
        # Sort the coordinates
        weights = torch.tensor([1e12, 1e8, 1e4, 1], device=input.device) 
        sortable_vals = (input.C * weights).sum(dim=1)
        sorted_coords_indices = sortable_vals.argsort()

        input = ME.SparseTensor(
            features=input.F[sorted_coords_indices],
            coordinates=input.C[sorted_coords_indices],
            tensor_stride=input.tensor_stride,
            device=input.device
        )

        output = super().forward(input)
        
        # Sort the coordinates
        weights = torch.tensor([1e12, 1e8, 1e4, 1], device=input.device) 
        sortable_vals = (output.C * weights).sum(dim=1)
        sorted_coords_indices = sortable_vals.argsort()

        output = ME.SparseTensor(
            features=output.F[sorted_coords_indices],
            coordinates=output.C[sorted_coords_indices],
            tensor_stride=output.tensor_stride,
            device=output.device
        )

        return output


class SortedMinkowskiGenerativeConvolutionTranspose(ME.MinkowskiGenerativeConvolutionTranspose):
    """
    A Sorted MinkowskiConvolution to allow deterministic computation on the same hardware
    """
    def forward(self, input):
        # Sort the coordinates
        weights = torch.tensor([1e12, 1e8, 1e4, 1], device=input.device) 
        sortable_vals = (input.C * weights).sum(dim=1)
        sorted_coords_indices = sortable_vals.argsort()

        input = ME.SparseTensor(
            features=input.F[sorted_coords_indices],
            coordinates=input.C[sorted_coords_indices],
            tensor_stride=input.tensor_stride,
            device=input.device
        )

        output = super().forward(input)
        
        # Sort the coordinates
        weights = torch.tensor([1e12, 1e8, 1e4, 1], device=output.device) 
        sortable_vals = (output.C * weights).sum(dim=1)
        sorted_coords_indices = sortable_vals.argsort()

        output = ME.SparseTensor(
            features=output.F[sorted_coords_indices],
            coordinates=output.C[sorted_coords_indices],
            tensor_stride=output.tensor_stride,
            device=output.device
        )

        return output

class SortedMinkowskiLeakyReLU(ME.MinkowskiLeakyReLU):
    """
    A Sorted MinkowskiLeakyReLU implementation to allow deterministic computation on the same hardware
    """
    def forward(self, input):
        # Sort the coordinates
        weights = torch.tensor([1e12, 1e8, 1e4, 1], device=input.device) 
        sortable_vals = (input.C * weights).sum(dim=1)
        sorted_coords_indices = sortable_vals.argsort()

        input = ME.SparseTensor(
            features=input.F[sorted_coords_indices],
            coordinates=input.C[sorted_coords_indices],
            tensor_stride=input.tensor_stride,
            device=input.device
        )

        output = super().forward(input)

        # Sort the coordinates
        weights = torch.tensor([1e12, 1e8, 1e4, 1], device=output.device) 
        sortable_vals = (output.C * weights).sum(dim=1)
        sorted_coords_indices = sortable_vals.argsort()

        output = ME.SparseTensor(
            features=output.F[sorted_coords_indices],
            coordinates=output.C[sorted_coords_indices],
            tensor_stride=output.tensor_stride,
            device=output.device
        )

        return output

class MeanScaleHyperprior(CompressionModel):
    """
    Mean-Scale Hyperprior Model employing an addaptive Gaussian Bottleneck.

    The Adaptive Gaussian Bottleneck is an extension of the Bottleneck presented in:
        Kamisli, Fatih, Fabien Racapé, and Hyomin Choi. "Variable-Rate Learned Image Compression with Multi-Objective 
        Optimization and Quantization-Reconstruction Offsets." 2024 Data Compression Conference (DCC). IEEE, 2024.
    based on the implementation in
        https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/models/vbr.py
    """
    def __init__(self, config):
        """
        Paramters:
            config (dict): Configurations for the model
                - C_bottleneck (int): Number of channels in the bottleneck
                - C_hyper_bottleneck (int): Number of channels in the hyper bottleneck
                - inverse_rescaling (bool): Use inverse rescaling
                - quantization_mode (string): Quantization mode (options: )
                - entropy_bottleneck_vbr (bool): Use VBR hyper bottleneck
                - adaptive_BN (bool): Use adaptive bottleneck
                - quantization_offset (bool): Use quantization offsets

        """
        super().__init__()
        C_bottleneck = config["C_bottleneck"]
        C_hyper_bottleneck = config["C_hyper_bottleneck"]
        self.inverse_rescaling = config["inverse_rescaling"]
        self.quantization_mode = config["quantization_mode"]
        self.entropy_bottleneck_vbr = config["entropy_bottleneck_vbr"]
        self.adaptive_BN = config["adaptive_BN"] if "adaptive_BN" in config.keys() else True
        
        self.eps = 0.0001
        self.quantization_offset = config["quantization_offset"]
        self.gaussian_conditional = GaussianConditional(None)

        if self.entropy_bottleneck_vbr:
            self.entropy_bottleneck = EntropyBottleneckVbr(C_hyper_bottleneck)
            self.gain2zqstep = nn.Sequential(
                nn.Linear(2, 10),            
                nn.ReLU(),
                nn.Linear(10, 10),            
                nn.ReLU(),
                nn.Linear(10, 1),            
                nn.Softplus(),
            )
            self.lower_bound_zqstep = LowerBound(0.5)
        else:
            self.entropy_bottleneck = EntropyBottleneck(C_hyper_bottleneck)

        self.h_a = nn.Sequential(
            ME.MinkowskiConvolution( in_channels=C_bottleneck, out_channels=C_hyper_bottleneck, kernel_size=3, dimension=3), 
            ME.MinkowskiLeakyReLU(inplace=False),
            ME.MinkowskiConvolution( in_channels=C_hyper_bottleneck, out_channels=C_hyper_bottleneck, kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiLeakyReLU(inplace=False),
            ME.MinkowskiConvolution( in_channels=C_hyper_bottleneck, out_channels=C_hyper_bottleneck, kernel_size=3, stride=2, dimension=3)
        )

        self.h_s = nn.Sequential(
            SortedMinkowskiGenerativeConvolutionTranspose( in_channels=C_hyper_bottleneck, out_channels=C_hyper_bottleneck, kernel_size=2, stride=2, bias=True, dimension=3),
            SortedMinkowskiLeakyReLU(inplace=False),
            SortedMinkowskiGenerativeConvolutionTranspose( in_channels=C_hyper_bottleneck, out_channels=C_bottleneck*3//2, kernel_size=2, stride=2, bias=True, dimension=3),
            SortedMinkowskiLeakyReLU(inplace=False),
            SortedMinkowskiConvolution( in_channels=C_bottleneck*3//2, out_channels=C_bottleneck*2, kernel_size=3, stride=1, dimension=3, bias=True)
        )

        self.scale_nn = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, C_bottleneck//4),
            nn.ReLU(),
            nn.Linear(C_bottleneck//4, C_bottleneck),
            nn.Softplus(),
        )
        self.rescale_nn = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, C_bottleneck//4),
            nn.ReLU(),
            nn.Linear(C_bottleneck//4, C_bottleneck),
            nn.Softplus(),
        )
        self.quant_nn = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )


    def get_offsets(self, stddev, scale):
        """
        Helper function to get offsets from stddevs and scale 

        Parameters:
            stddev (torch.tensor): 
                Tensor containing offsets
            scale (torch.tensor): 
                Tensor containing scales
        Returns:
            offsets (torch.tensor): 
                Offsets per element 
        """
        combined_inputs = torch.cat((scale.unsqueeze(dim=3), stddev.unsqueeze(dim=3)), dim=3)  
        offsets = self.quant_nn(combined_inputs).squeeze(dim=3)
        return offsets


    def forward(self, y, q):
        """
        Forward pass for training. 
        Parameters:
            y (ME.SparseTensor): Sparse Tensor containing the features
            q (torch.tensor): Quality values for the batch
        """
        z = self.h_a(y)

        z_feats = z.F.t().unsqueeze(0)
        y_feats = y.F.t().unsqueeze(0)

        y_batch_indices = y.C[:, 0]
        
        # Adaptive Bottleneck Computation
        if self.adaptive_BN:
            scale = self.scale_nn(q) + self.eps
            if self.entropy_bottleneck_vbr:
                z_qstep = self.gain2zqstep(q)
            
            scale = scale[y_batch_indices].t().unsqueeze(0)
            if self.inverse_rescaling:
                # Rescale is the reciprocal of scale
                rescale = torch.tensor(1.0) / scale.clone().detach() 
            else:
                # Using a NN to compute the rescaling factors
                rescale = torch.tensor(1.0) / self.rescale_nn(q)
                rescale = rescale[y_batch_indices].t().unsqueeze(0)
        else: 
            N, F = y.C.shape[0], y.F.shape[1]
            scale = torch.ones((1,F,N), device=y.device)
            rescale = torch.ones((1,F,N), device=y.device)


        if self.quantization_mode == "uniform":
            # Additive Uniform Noise Quantization Proxy
            z_hat, z_likelihoods = self.entropy_bottleneck(z_feats)
        else:
            # Straight-Through Quantization Proxy
            if self.entropy_bottleneck_vbr:
                # Entropy Bottleneck VBR currently only supported for training, not testing 
                z_qstep = self.lower_bound_zqstep(z_qstep)
                z_qstep = z_qstep[y_batch_indices].t().unsqueeze(0)
                z_hat, z_likelihoods = self.entropy_bottleneck(z_feats, qs=z_qstep, training=None, ste=True)
            else:
                # Straight through quantization
                _, z_likelihoods = self.entropy_bottleneck(z_feats)
                z_offset = self.entropy_bottleneck._get_medians()
                z_feats = z_feats - z_offset
                z_hat = ops.quantize_ste(z_feats) + z_offset

        z_hat = ME.SparseTensor(features=z_hat[0].t(), 
                                coordinates=z.C,
                                tensor_stride=32,
                                device=z.device)

        # Estimate the Parameters for the Gaussian Entropy Model
        gaussian_params = self.h_s(z_hat)
        gaussian_params_feats = gaussian_params.features_at_coordinates(y.C.float())
        scales_hat, means_hat = gaussian_params_feats.chunk(2, dim=1)
        scales_hat = scales_hat.t().unsqueeze(0)
        means_hat = means_hat.t().unsqueeze(0) 

        # Gaussian Entropy Model
        if self.quantization_offset:
            # Uses quantization offsets
            y_feats_tmp = scale * (y_feats - means_hat)
            signs = torch.sign(y_feats_tmp).detach()

            if self.quantization_mode == "uniform":
                # Additive Uniform Noise Quantization Proxy
                y_q_abs = quantize_noise(torch.abs(y_feats_tmp))
            else:
                # Straight-Through Quantization Proxy
                y_q_abs = ops.quantize_ste(torch.abs(y_feats_tmp))

            _, y_likelihoods = self.gaussian_conditional(
                y_feats * scale,
                scales_hat * scale,
                means= means_hat * scale
            )

            # Computation of quantization offsets
            y_q_stdev = self.gaussian_conditional.lower_bound_scale(scales_hat * scale)
            q_offsets = (-1) * self.get_offsets(y_q_stdev, scale.clone().detach())
            q_offsets[y_q_abs < 0.0001] = (0)

            y_hat = signs * (y_q_abs + q_offsets)
            y_hat = y_hat * rescale + means_hat
        else:
            # No quantization offsets
            y_hat, y_likelihoods = self.gaussian_conditional(
                y_feats * scale,
                scales_hat * scale,
                means=means_hat * scale
            )
            y_hat = y_hat * rescale


        y_hat = ME.SparseTensor(features=y_hat[0].t(), 
                                coordinates=y.C,
                                tensor_stride=8,
                                device=y.device)

        return y_hat, (y_likelihoods, z_likelihoods)



    def compress(self, y, q):
        """
        Compress a latent representation y using the given quality configuration q
        Parameters:
            y (ME.SparseTensor): 
                Sparse Tensor containing the features
            q (torch.tensor): 
                Quality values for the batch
        Returns:
            points (list): 
                List of base geometry for hyprior z and features y
            strings (list): 
                List of entropy coded representation (of hyperprior z, latent features y)
            shapes (int): 
                Shape of the hyperprior z (# Num Features)
        """
        # Hyper analysis
        z = self.h_a(y)

        # Sort points
        y = sort_tensor(y)
        z = sort_tensor(z)

        y_batch_indices = y.C[:, 0]
        shape = [z.F.shape[0]]

        # Hyperprior Compression
        z_strings = self.entropy_bottleneck.compress(z.F.t().unsqueeze(0))
        z_hat_feats = self.entropy_bottleneck.decompress(z_strings, shape)

        z_hat = ME.SparseTensor(features=z_hat_feats[0].t(), 
                                coordinates=z.C,
                                tensor_stride=32,
                                device=z.device)

        # Estimate Parameters for Gaussian Entropy model from z_hat
        gaussian_params = self.h_s(z_hat)
        gaussian_params_feats = gaussian_params.features_at_coordinates(y.C.float())
        scales_hat, means_hat = gaussian_params_feats.chunk(2, dim=1)
        scales_hat = scales_hat.t().unsqueeze(0)
        means_hat = means_hat.t().unsqueeze(0)

        if self.adaptive_BN:
            # Use adaptive Bottleneck
            scale = self.scale_nn(q) + self.eps
            scale = scale[y_batch_indices].t().unsqueeze(0)
        else:
            # non-adaptive bottleneck (emulated by always scaling with 1.0)
            N, F = y.C.shape[0], y.F.shape[1]
            scale = torch.ones((1,F,N), device=y.device)

        # Gaussian Entropy Model Compression
        indexes = self.gaussian_conditional.build_indexes(scales_hat * scale)
        y_strings = self.gaussian_conditional.compress(
            y.F.t().unsqueeze(0) * scale, 
            indexes, 
            means=means_hat * scale)

        y_points = y.C
        z_points = z.C
        points = [y_points, z_points]
        strings = [y_strings, z_strings]
        return points, strings, shape


    def decompress(self, points, strings, shape, q):
        """
        Decompress the bitstream to receive the latent representation y using the given quality configuration q.
        Parameters:
            points (list): 
                List of base geometry for hyprior z and features y
            strings (list): 
                List of entropy coded representation (of hyperprior z, latent features y)
            shapes (int): 
                Shape of the hyperprior z (# Num Features)
            q (torch.tensor): 
                Quality values for the batch
        Returns:
            y_hat (ME.SparseTensor): 
                Sparse Tensor containing the features after decompression
        """
        assert isinstance(strings, list) and len(strings) == 2
        assert isinstance(points, list) and len(points) == 2

        y_strings, z_strings = strings[0], strings[1]
        y_points, z_points = points[0], points[1]

        # Sort the points (required to have same order as for compression)
        y_points = sort_points(y_points)
        z_points = sort_points(z_points)

        y_batch_indices = y_points[:, 0].int()

        # Hyper-Prior decompression
        z_hat_feats = self.entropy_bottleneck.decompress(z_strings, shape)
        z_hat = ME.SparseTensor(features=z_hat_feats[0].t(),
                                coordinates=z_points,
                                tensor_stride=32,
                                device=z_points.device)

        # Estimation of the Parameters for the Gaussian Entropy Model 
        gaussian_params = self.h_s(z_hat)
        gaussian_params_feats = gaussian_params.features_at_coordinates(y_points.float())
        scales_hat, means_hat = gaussian_params_feats.chunk(2, dim=1)
        scales_hat = scales_hat.t().unsqueeze(0)
        means_hat = means_hat.t().unsqueeze(0)

        if self.adaptive_BN:
            # Use the adaptive Bottleneck
            scale = self.scale_nn(q) + self.eps
            scale = scale[y_batch_indices].t().unsqueeze(0)
            if self.inverse_rescaling:
                # Rescaling through reciprocal
                rescale = torch.tensor(1.0) / scale
            else:
                # Rescaling through a seperate NN
                rescale = torch.tensor(1.0) / self.rescale_nn(q) 
                rescale = rescale[y_batch_indices].t().unsqueeze(0)
        else:
            # Emulate non-adaptive bottleneck through scales/rescale set to 1.0
            scale = torch.ones_like(means_hat, device=z_points.device)
            rescale = torch.ones_like(means_hat, device=z_points.device)

        # Gaussian Entropy Model
        indexes = self.gaussian_conditional.build_indexes(scales_hat * scale)
        if self.quantization_offset:
            # Use quantization offsets for Decompression
            q_val = self.gaussian_conditional.decompress(y_strings, indexes)
            q_abs, signs = q_val.abs(), torch.sign(q_val)

            # Compute offsets
            y_q_stdev = self.gaussian_conditional.lower_bound_scale(scales_hat * scale)
            q_offsets = (-1) * self.get_offsets(y_q_stdev, scale)
            q_offsets[q_abs < 0.0001] = (0)
        
            # Rescale to y_hat
            y_hat = signs * (q_abs + q_offsets)
            y_hat = y_hat * rescale + means_hat
        else:
            # No quantization offsets
            y_hat = self.gaussian_conditional.decompress(y_strings, indexes, means=means_hat * scale)

        y_hat = ME.SparseTensor(features=y_hat[0].t(),
                                coordinates=y_points,
                                tensor_stride=8,
                                device=y_points.device)
        return y_hat