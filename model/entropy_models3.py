import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from utils import sort_points, sort_tensor

from compressai.entropy_models import EntropyBottleneck, GaussianConditional 
from compressai.models.base import CompressionModel
from compressai.ops import ops

class SortedMinkowskiConvolution(ME.MinkowskiConvolution):
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

class MeanScaleHyperprior_Map(CompressionModel):
    def __init__(self, config):
        """
        Paramters
        ---------
            C_bottleneck: int
                Number of channels in the bottlneck
            C_hyper_bottlneck: int
                Number of channels in the bottlneck of the hyperprior model
            N: int
                Number of channels in between
        """
        super().__init__()
        C_bottleneck = config["C_bottleneck"]
        C_hyper_bottleneck = config["C_hyper_bottleneck"]
        self.inverse_rescaling = config["inverse_rescaling"]
        self.quantization_mode = config["quantization_mode"]
        
        self.eps = 0.01
        self.quantization_offset = config["quantization_offset"]
        self.entropy_bottleneck = EntropyBottleneck(C_hyper_bottleneck)
        self.gaussian_conditional = GaussianConditional(None)

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

        # Compute scales per channel from q values
        #self.scale_nn = nn.Sequential(
        #    nn.Linear(2, C_bottleneck//4),
        #    nn.ReLU(),
        #    nn.Linear(C_bottleneck//4, C_bottleneck),
        #    nn.ReLU(),
        #    nn.Linear(C_bottleneck, C_bottleneck),
        #    nn.Softplus(),
        #)
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
        # Compute Quantization offsets from gain and stdev
        self.quant_nn = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )


    def scale_per_batch(self, x, scale, batch_indices):
        """
        Scale tensor x per batch
        batch_indices : N
        x: 1, C, N
        scale: B, C
        """
        y = x.clone()
        num_batches = int(torch.max(batch_indices) + 1)
        for i in range(num_batches):
            mask = (batch_indices == i)

            result = scale[i].unsqueeze(1) * x[0, :, mask]
            y[0, :, mask] = result
        return y

    def get_offsets(self, stddev):
        """
        Scale tensor x per batch
        batch_indices : N
        x: 1, C, N
        scale: B, C
        """
        result = self.quant_nn(stddev.unsqueeze(dim=3)).squeeze(3)

        return result

    def forward(self, y, q):
        # Scaling
        y_batch_indices = y.C[:, 0]
        scale = self.scale_nn(q)
        scale = scale[y_batch_indices].t().unsqueeze(0)
        if self.inverse_rescaling:
            rescale = torch.tensor(1.0) / scale.clone().detach() #TODO: is this a good idea?
        else:
            rescale = torch.tensor(1.0) / self.rescale_nn(q)
            rescale = rescale[y_batch_indices].t().unsqueeze(0)
        y = ME.SparseTensor(features=y.F * scale[0].T, 
                                coordinates=y.C,
                                tensor_stride=8,
                                device=y.device)

        z = self.h_a(y)

        z_feats = z.F.t().unsqueeze(0)
        y_feats = y.F.t().unsqueeze(0)

        # Hyperpriors
        _, z_likelihoods = self.entropy_bottleneck(z_feats)
        z_offset = self.entropy_bottleneck._get_medians()
        z_feats = z_feats - z_offset
        z_hat = ops.quantize_ste(z_feats) + z_offset

        # Reconstruct z_hat
        z_hat = ME.SparseTensor(features=z_hat[0].t(), 
                                coordinates=z.C,
                                tensor_stride=32,
                                device=z.device)

        # Hyper synthesis
        gaussian_params = self.h_s(z_hat)
        gaussian_params_feats = gaussian_params.features_at_coordinates(y.C.float())
        scales_hat, means_hat = gaussian_params_feats.chunk(2, dim=1)
        scales_hat = scales_hat.t().unsqueeze(0)
        means_hat = means_hat.t().unsqueeze(0) 

        # Find the right scales
        y_feats_tmp = y_feats - means_hat
        signs = torch.sign(y_feats_tmp).detach()
        y_q_abs = ops.quantize_ste(torch.abs(y_feats_tmp))

        y_q_stdev = self.gaussian_conditional.lower_bound_scale(scale)

        # Get offsets from stdev and scale (Per element)
        q_offsets = (-1) * self.get_offsets(y_q_stdev)
        q_offsets[y_q_abs < 0.0001] = (0)

        y_hat = signs * (y_q_abs + q_offsets)
        # Rescaling
        y_hat = rescale * (y_hat + means_hat)

        y_hat = ME.SparseTensor(features=y_hat[0].t(), 
                                coordinates=y.C,
                                tensor_stride=8,
                                device=y.device)

        _, y_likelihoods = self.gaussian_conditional(
            y_feats,
            scales_hat,
            means=means_hat
        )

        return y_hat, None, (y_likelihoods, z_likelihoods)



    def compress(self, y, q):
        # Scaling
        y_batch_indices = y.C[:, 0]
        scale = self.scale_nn(q)
        scale = scale[y_batch_indices].t().unsqueeze(0)

        y = ME.SparseTensor(features=y.F * scale[0].T, 
                                coordinates=y.C,
                                tensor_stride=8,
                                device=y.device)
        # Hyper analysis
        z = self.h_a(y)

        # Sort points
        y = sort_tensor(y)
        z = sort_tensor(z)

        # Entropy Bottleneck
        shape = [z.F.shape[0]]
        z_strings = self.entropy_bottleneck.compress(z.F.t().unsqueeze(0))
        z_hat_feats = self.entropy_bottleneck.decompress(z_strings, shape)

        # Reconstruct z_hat
        z_hat = ME.SparseTensor(features=z_hat_feats[0].t(), 
                                coordinates=z.C,
                                tensor_stride=32,
                                device=z.device)

        gaussian_params = self.h_s(z_hat)
        gaussian_params_feats = gaussian_params.features_at_coordinates(y.C.float())
        scales_hat, means_hat = gaussian_params_feats.chunk(2, dim=1)
        scales_hat = scales_hat.t().unsqueeze(0)
        means_hat = means_hat.t().unsqueeze(0)

        indexes = self.gaussian_conditional.build_indexes(scale)
        y_strings = self.gaussian_conditional.compress(
            y.F.t().unsqueeze(0), 
            indexes, 
            means=means_hat)

        # Points are needed, to be compressed later
        y_points = y.C
        z_points = z.C

        # Pack it
        points = [y_points, z_points]
        strings = [y_strings, z_strings]
        return points, strings, shape


    def decompress(self, points, strings, shape, q):
        assert isinstance(strings, list) and len(strings) == 2

        # Get the points back
        y_points, z_points = points[0], points[1]
        y_points = sort_points(y_points)
        z_points = sort_points(z_points)
        y_strings, z_strings = strings[0], strings[1]

        y_batch_indices = y_points[:, 0]

        z_hat_feats = self.entropy_bottleneck.decompress(z_strings, shape)
        z_hat = ME.SparseTensor(features=z_hat_feats[0].t(),
                                coordinates=z_points,
                                tensor_stride=32,
                                device=z_points.device)
        # Decompress y_hat
        gaussian_params = self.h_s(z_hat)
        gaussian_params_feats = gaussian_params.features_at_coordinates(y_points.float())

        scales_hat, means_hat = gaussian_params_feats.chunk(2, dim=1)
        scales_hat = scales_hat.t().unsqueeze(0)
        means_hat = means_hat.t().unsqueeze(0)


        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        q_val = self.gaussian_conditional.decompress(y_strings, indexes)
        q_abs, signs = q_val.abs(), torch.sign(q_val)

        #Quantization offset
        y_q_stdev = self.gaussian_conditional.lower_bound_scale(scales_hat)

        # Get offsets from stdev and scale (Per element)
        q_offsets = (-1) * self.get_offsets(y_q_stdev)
        q_offsets[q_abs < 0.0001] = (0)
        
        y_hat = signs * (q_abs + q_offsets)
        y_hat = y_hat + means_hat

        # Rescaling
        rescale = torch.tensor(1.0) / self.rescale_nn(q) 
        rescale = rescale[y_batch_indices].t().unsqueeze(0)
        y_hat = y_hat * rescale
        y_hat = ME.SparseTensor(features=y_hat[0].t(),
                                coordinates=y_points,
                                tensor_stride=8,
                                device=y_points.device)
        return y_hat


def quantize_noise(x):
    half = float(0.5)
    noise = torch.empty_like(x).uniform_(-half, half)
    x = x + noise
    return x