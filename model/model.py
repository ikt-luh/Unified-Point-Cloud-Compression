import torch
import numpy as np
import MinkowskiEngine as ME
from compressai.models.base import CompressionModel

import os
import subprocess
import open3d as o3d
from bitstream import BitStream

from .entropy_models2 import *
#from .entropy_models3 import *
from .transforms3 import *

import utils

class ColorModel(CompressionModel):
    def __init__(self, config):
        super().__init__()

        self.g_a = AnalysisTransform(config["g_a"])
        self.g_s = SparseSynthesisTransform(config["g_s"])

        self.entropy_model = MeanScaleHyperprior_Map(config["entropy_model"])


    def update(self):
        """
        Update the scale tables of entropy models
        """
        self.entropy_model.update(force=True)



    def aux_loss(self):
        """
        Get the aux loss of the entropy model
        """
        return self.entropy_model.aux_loss()
    


    def forward(self, x, Q, Lambda):
        """
        Parameters
        ----------
        x : ME.SparseTensor
            Input Tensor of geometry and features
        """
        # Save coords for decoding
        coords = ME.SparseTensor(coordinates=x.C.clone(),
                                 features=torch.ones(x.C.shape[0], 1),
                                 device=x.device)

        # Pad input tensor
        x = ME.SparseTensor(coordinates=x.C.clone(),
                            features=torch.cat([torch.ones((x.C.shape[0], 1), device=x.device), x.F], dim=1))

        # Analysis Transform
        y, Q, k = self.g_a(x, Q)

        # Entropy Bottleneck
        #y_hat, Q_hat, likelihoods = self.entropy_model(y)
        y_hat, Q_hat, likelihoods = self.entropy_model(y, Q)

        # Split coords after entropy coding
        likelihoods = {"y": likelihoods[0], "z": likelihoods[1]}


        # Synthesis Transform(s)
        ####  x_hat, points, predictions = self.g_s(y_hat, Q_hat, coords=coords, k=k)
        x_hat, points, predictions = self.g_s(y_hat, Q, coords=coords, k=k)
        
        # Building Output dictionaries
        output = {
            "prediction": x_hat,
            "points": points,
            "occ_predictions": predictions,
            "q_map": Lambda,
            "likelihoods": likelihoods
        }

        return output



    def compress(self, x, Q, path=None, block_size=1024, scaling_factor=1.0):
        """
        Compress a point cloud with optional block partitioning and downscaling.
    
        Parameters
        ----------
        x: torch.tensor, shape Nx6
            Tensor containing the point cloud, N is the number of points.
            Coordinates as first 3 dimensions, colors as last 3
        Q: Quantization parameter or data structure
        path: str (default=None)
            Path to store the binaries to. If None, the compression is mocked.
        block_size: int (default=100000)
            Maximum number of points per block. If N > block_size, point cloud is partitioned.
        scaling_factor: float (default=1.0)
            Factor to downscale coordinates during compression (1.0 means no scaling).
    
        Returns
        -------
        strings: list
            List of strings (bitstreams), only returned if path=None
        shape: list
            List of shapes, only returned if path=None
        """
        N = x.shape[0]

        # Apply downscaling to coordinates if scaling_factor != 1.0
        if scaling_factor != 1.0:
            x[:, :3] = torch.round(x[:, :3] / scaling_factor).int()

        min_coords = torch.min(x[:, :3], dim=0)[0]

        block_indices = ((x[:, :3] - min_coords) / block_size).floor().int()

        # Sort points by block indices (to group points from the same block together)
        sorted_indices = torch.argsort(block_indices[:, 0] * 1e6 + block_indices[:, 1] * 1e3 + block_indices[:, 2])
        x_sorted = x[sorted_indices]
        block_indices_sorted = block_indices[sorted_indices]

        # Find unique block indices and their starting positions
        unique_blocks, unique_indices, counts = torch.unique_consecutive(block_indices_sorted, dim=0, return_inverse=True, return_counts=True)

        # Split the point cloud into blocks if it exceeds the block_size
        bitstreams = []
        block_shapes = []
        block_coordinates = []
        block_q_vals = []
        block_k = []

        start_idx = 0
        for i, count in enumerate(counts.tolist()):
            end_idx = start_idx + count
            x_block = x_sorted[start_idx:end_idx]

            # Build input point cloud from tensors
            batch_vec = torch.zeros((x_block.shape[0], 1), device=x_block.device)
            points = torch.cat([batch_vec, x_block[:, :3].contiguous()], dim=1).to(x_block.device)
            colors = x_block[:, 3:6].contiguous().to(x_block.device)

            # Minkowski Tensor
            input_block = ME.SparseTensor(coordinates=points,
                                        features=colors,
                                        device=x_block.device)
            coords, feats = ME.utils.sparse_quantize(
                coordinates=input_block.C,
                features=input_block.F,
                quantization_size=1.0
            )

            input_block = ME.SparseTensor(coordinates=coords.int(),
                                        features=torch.cat([torch.ones((coords.shape[0], 1), device=x_block.device), feats], dim=1),
                                        device=x_block.device)
            

            # Analysis Transform
            y, q_vals, k = self.g_a(input_block, Q)

            # Entropy Bottleneck Compression
            #_ , strings, shape = self.entropy_model.compress(y)        

            # MOO Compression
            print(Q)
            _ , strings, shape = self.entropy_model.compress(y, Q)
            block_q_vals.append(Q)

            block_coordinates.append(y.C)  # Save block coordinates
            block_shapes.append(shape)
            block_k.append(k)
            bitstreams.append(strings)
            start_idx = end_idx

        # Combine block results into one bitstream (for saving)
        if path:
            self.save_bitstream(path=path, blocks_coordinates=block_coordinates, blocks_strings=bitstreams, blocks_shapes=block_shapes, blocks_k=block_k, blocks_q=block_q_vals)
            #self.save_bitstream(path=path, blocks_coordinates=block_coordinates, blocks_strings=bitstreams, blocks_shapes=block_shapes, blocks_k=block_k)
        else:
            return bitstreams, block_shapes, block_k, block_coordinates, block_q_vals



    def decompress(self, path=None, coordinates=None, strings=None, shape=None, k=None, q_vals=None):
        """
        Decompress a point cloud bitstream.
    
        Parameters
        ----------
        path: str
            Path of the point cloud bitstream.
        coordinates: torch.tensor
            Point Cloud geometry required to decode the attributes.
        strings: list
            List of strings (bitstreams), only returned if path=None.
        shape: list
            List of shapes, only returned if path=None.
        k: list
            Number of points at each stage.
    
        Returns
        -------
        x_hat: torch.tensor, Nx6
            Decompressed and reconstructed point cloud.
        """
        device = self.g_s.down_conv.kernel.device
        if path:
            coordinates, strings, shape, k, q_vals = self.load_bitstream(path)
            #coordinates, strings, shape, k = self.load_bitstream(path)
            for i, _ in enumerate(coordinates):
                block_coords = coordinates[i].to(device) 
                batch_vec = torch.zeros((block_coords.shape[0], 1), device=block_coords.device)
                coordinates[i] = torch.cat([batch_vec, block_coords.contiguous()], dim=1)
                q_vals[i] = q_vals[i].to(device) 
                #q_vals = []

        # Decompress blocks
        x_hat_list = []
        for i, (block_strings, block_shape, block_coords, block_k) in enumerate(zip(strings, shape, coordinates, k)):
            # Perform entropy decoding and synthesis transform

            latent_coordinates_2 = ME.SparseTensor(coordinates=block_coords.clone(), features=torch.ones((block_coords.shape[0], 1)), tensor_stride=8, device=block_coords.device)
            latent_coordinates_2 = self.g_s.down_conv(latent_coordinates_2)
            latent_coordinates_2 = self.g_s.down_conv(latent_coordinates_2)
            points = [block_coords, latent_coordinates_2.C]

            #y_hat, Q_hat = self.entropy_model.decompress(points, block_strings, block_shape)
            y_hat = self.entropy_model.decompress(points, block_strings, block_shape, q_vals[i])
        
            # Perform synthesis
            if not len(q_vals)==0:
                Q_hat = q_vals[i]
            x_hat = self.g_s(y_hat, Q_hat, k=block_k)
            x_hat_list.append(x_hat)
    
        # Concatenate results
        features_list = [x_hat.F for x_hat in x_hat_list]
        coords_list = [x_hat.C for x_hat in x_hat_list]

        features_concat = torch.cat(features_list, dim=0)
        coords_concat = torch.cat(coords_list, dim=0)

        features_processed = torch.clamp(torch.round(features_concat * 255), 0.0, 255.0) / 255
        x_hat = torch.cat([coords_concat[:, 1:4], features_processed], dim=1)
        return x_hat


    #def save_bitstream(self, path, blocks_coordinates, blocks_strings, blocks_shapes, blocks_k):
    def save_bitstream(self, path, blocks_coordinates, blocks_strings, blocks_shapes, blocks_k, blocks_q):
        """
        Save multiple blocks to a bitstream.

        Parameters
        ----------
        path: str
            Path to store the data.
        blocks_coordinates: list of torch.tensor
            List of coordinates for each block.
        blocks_strings: list of lists
            List of bitstreams corresponding to each block.
        blocks_shapes: list of lists
            List of shapes for each block's feature representation.
        blocks_k: list of lists
            Number of points at each stage for each block.
        """
        stream = BitStream()

        # Write the number of blocks
        num_blocks = len(blocks_coordinates)
        stream.write(num_blocks, np.int32)

        # For each block, save the block's data using the existing save_bitstream logic
        for i in range(num_blocks):
            points = blocks_coordinates[i]
            strings = blocks_strings[i]
            shape = blocks_shapes[i]
            k = blocks_k[i]
            q = blocks_q[i]

            # Save each block using the original single-block method
            # Encode points with G-PCC and write the shape, points bitstream, strings, and k values
            points_bitstream = self.gpcc_encode(points, path)

            ## Write block header
            # Shape
            stream.write(shape, np.int32)
            stream.write(len(points_bitstream), np.int32)
            stream.write(q[0, 0].cpu(), np.float64)
            stream.write(q[0, 1].cpu(), np.float64)

            # String lengths
            for string in strings:
                stream.write(len(string[0]), np.int32)
            for ks in k:
                stream.write(ks, np.int32)

            # Write block content
            stream.write(points_bitstream)
            for string in strings:
                stream.write(string[0])


        # Write final bitstream to file
        bit_string = stream.__str__()
        byte_array = bytes(int(bit_string[i:i+8], 2) for i in range(0, len(bit_string), 8))

        with open(path, "wb") as binary:
            binary.write(byte_array)
    

    def load_bitstream(self, path):
        """
        Load the bitstream from disk for multiple blocks.

        Parameters
        ----------
        path: str
            Path to the bitstream file.

        Returns
        -------
        blocks_coordinates: list
            List of coordinates for each block.
        blocks_strings: list
            List of bitstreams for each block.
        blocks_shapes: list
            Shapes of feature representations for each block.
        blocks_k: list
            Number of points at each stage for each block.
        """
        # Initialize BitStream and load the file
        stream = BitStream()
        with open(path, "rb") as binary:
            data = binary.read()

        stream.write(data, bytes)

        # Read the number of blocks
        num_blocks = stream.read(np.int32)

        # Initialize lists to store the block-wise data
        blocks_coordinates = []
        blocks_strings = []
        blocks_shapes = []
        blocks_k = []
        blocks_q = []

        # For each block, load the block's data using the existing load_bitstream logic
        for _ in range(num_blocks):
            # Read the block header
            shape = [int(stream.read(np.uint32))]
            len_points_bitstream = stream.read(np.uint32)
            q_vals = torch.tensor([stream.read(np.float64), stream.read(np.float64)]).reshape(1,2)
            len_string_1 = stream.read(np.uint32)
            len_string_2 = stream.read(np.uint32)
            string_lengths = [len_string_1, len_string_2]
            k1 = [stream.read(np.uint32)]
            k2 = [stream.read(np.uint32)]
            k3 = [stream.read(np.uint32)]
            k = [k1, k2, k3]

            # Read the block payload
            points_bitstream = stream.read(int(len_points_bitstream) * 8)

            strings = []
            for i in range(2):  # Assuming two bitstreams for each block
                string = stream.read(int(string_lengths[i]) * 8)
                bit_string = string.__str__()
                byte_string = bytes(int(bit_string[j:j + 8], 2) for j in range(0, len(bit_string), 8))
                strings.append([byte_string])

            # Decode the points using G-PCC
            block_coordinates = self.gpcc_decode(points_bitstream, path)

            # Append block data to lists
            blocks_coordinates.append(block_coordinates)
            blocks_strings.append(strings)
            blocks_shapes.append(shape)
            blocks_k.append(k)
            blocks_q.append(q_vals)

        return blocks_coordinates, blocks_strings, blocks_shapes, blocks_k, blocks_q


    def gpcc_encode(self, points, directory):
        """
        Encode a list of points with G-PCC
        """
        directory, _ = os.path.split(directory)
        tmp_dir = os.path.join(directory, "points_enc.ply")
        bin_dir = os.path.join(directory, "points_enc.bin")

        # Save points as ply
        dtype = o3d.core.float32
        p_tensor = o3d.core.Tensor(points.detach().cpu().numpy()[:, 1:], dtype=dtype)
        pc = o3d.t.geometry.PointCloud(p_tensor)
        o3d.t.io.write_point_cloud(tmp_dir, pc, write_ascii=True)

        # G-PCC
        subp=subprocess.Popen('./dependencies/mpeg-pcc-tmc13/build/tmc3/tmc3'+ 
                                ' --mode=0' + 
                                ' --positionQuantizationScale=1' + 
                                ' --trisoupNodeSizeLog2=0' + 
                                ' --neighbourAvailBoundaryLog2=8' + 
                                ' --intra_pred_max_node_size_log2=6' + 
                                ' --inferredDirectCodingMode=0' + 
                                ' --maxNumQtBtBeforeOt=4' +
                                ' --uncompressedDataPath='+tmp_dir + 
                                ' --compressedStreamPath='+bin_dir, 
                                shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Read stdout and stderr
        stdout, stderr = subp.communicate()

        # Print the outputs
        if subp.returncode != 0:
            print("Error occurred:")
            print(stderr.decode())
            c=subp.stdout.readline()

        # Read the bytes to return
        with open(bin_dir, "rb") as binary:
            data = binary.read()
        
        # Clean up
        os.remove(tmp_dir)
        os.remove(bin_dir)

        return data



    def gpcc_decode(self, bin, directory):
        directory, _ = os.path.split(directory)
        tmp_dir = os.path.join(directory, "points_dec.ply")
        bin_dir = os.path.join(directory, "points_dec.bin")
        
        # Write to file
        bit_string = bin.__str__()
        byte_array = bytes(int(bit_string[i:i+8], 2) for i in range(0, len(bit_string), 8))

        with open(bin_dir, "wb") as binary:
            binary.write(byte_array)
        subp=subprocess.Popen('./dependencies/mpeg-pcc-tmc13/build/tmc3/tmc3'+ 
                                ' --mode=1'+ 
                                ' --compressedStreamPath='+bin_dir+ 
                                ' --reconstructedDataPath='+tmp_dir+
                                ' --outputBinaryPly=0',
                                shell=True, stdout=subprocess.PIPE)
        c=subp.stdout.readline()
        while c:
            c=subp.stdout.readline()
            #print(c)
    
        # Load ply
        pcd = o3d.io.read_point_cloud(tmp_dir)
        points = torch.tensor(np.asarray(pcd.points))

        # Clean up
        os.remove(tmp_dir)
        os.remove(bin_dir)
        return points
