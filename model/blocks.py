import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from compressai.layers import GDN


class MinkowskiGDN(GDN):
    """
    Extension of Generalized Divisive Normalization layer to Sparse Tensors
        Johnston, Nick, et al. "Computationally efficient neural image compression." arXiv preprint arXiv:1912.08771 (2019).
    Based on GDN1 implementation:
        https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/layers/gdn.py
    """
    def __init__(self,
        in_channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
        kernel_size: int = 1,
    ):
        super(MinkowskiGDN, self).__init__(in_channels, inverse, beta_min, gamma_init)
        self.kernel_size = kernel_size


    def forward(self, x):
        """
        Forward Pass for GDN / IGDN.

        Parameters:
            x (ME.SparseTensor):
                Activation Input
        
        Returns:
            x (ME.SparseTensor):
                Activation Output
        """
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
        output =  ME.SparseTensor(coordinates=x.C, 
                                  features=output_features, 
                                  tensor_stride=x.tensor_stride, 
                                  device=x.device)
        return output