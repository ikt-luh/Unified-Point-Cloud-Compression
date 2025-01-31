import os
import open3d as o3d
import torch
import subprocess
import MinkowskiEngine as ME 
import numpy as np
import time

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Source: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_bits(strings):
    """
    Computes the bpp for a nested array of strings
    
    Parameters:
        strings (list):
            Nested list of strings
    
    Returns:
        total_bits (int):
            Total bits required to save the nested list
    """
    total_bits = 0
    for string in strings:
        if not isinstance(string, list):
            total_bits += len(string) * 8
        else:
            total_bits += count_bits(string)
    return total_bits



def get_o3d_pointcloud(pc):
    """
    Generates a o3d point cloud on cpu from a torch tensor.

    Parameters:
        pc (torch.tensor):
            Tensor representing the point cloud

    Returns:
        o3d_pc (o3d.geometry.PointCloud):
            Point Cloud object in o3d
    """
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(pc[:, :3].cpu().numpy())
    o3d_pc.colors = o3d.utility.Vector3dVector(pc[:, 3:].cpu().numpy())
    return o3d_pc


def render_pointcloud(pc, path, point_size=1.0):
    """
    Render the point cloud from 6 views along x,y,z axis

    Parameters:
    pc (o3d.geometry.PointCloud):
        Point Cloud to be rendered
    path (str):
        Format String with a open key field for formatting
    """
    settings = {
        "front":  [[0, -1, 0], [0, 0, 1]],
        "back":   [[0, 1, 0], [0, 0, 1]],
        "left":   [[-1, 0, 0], [0, 0, 1]],
        "right":  [[1, 0, 0], [0, 0, 1]],
        "top":    [[0, 0, 1], [0, 1, 0]],
        "bottom": [[0, 0, -1], [0, 1, 0]]
    }

    # Path
    dir, _ = os.path.split(path)
    if not os.path.exists(dir):
        os.mkdir(dir)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis.add_geometry(pc)
    render_options = vis.get_render_option()
    render_options.point_size = 1.5  * point_size 

    for key, view in settings.items():
        view_control = vis.get_view_control()
        view_control.set_front(view[0])
        view_control.set_up(view[1])
        view_control.set_zoom(0.8)

        vis.update_renderer()
        
        image_path = path.format(key)
        vis.capture_screen_image(image_path, do_render=True)

    vis.destroy_window()


def downsampled_coordinates(coordinates, factor, batched=False):
    """
    Compute the remaining coordinates after downsampling by a factor
    TODO: MIGHT BE UNUSED, remove potentially

    Parameters
    coordinates: ME.SparseTensor
        Tensor containing the orignal coordinates
    factor: int
        Downsampling factor (mutliple of 2)

    returns
    coords: torch.tensor
        Unique coordinates of the tensor
    """
    # Handle torch tensors and ME.SparseTensors
    coords = coordinates if torch.is_tensor(coordinates) else coordinates.C

    if coords.shape[1] == 3:
        coords = torch.floor(coords / factor) * factor
    else:
        # Exclude batch id
        coords[:, 1:4] = torch.floor((coords[:, 1:4]) / factor) * factor

    coords = torch.unique(coords, dim=0) 
    return coords


def sort_tensor(sparse_tensor):
    """
    Sort the coordinates of a tensor

    Parameters:
        sparse_tensor (ME.SparseTensor):
            Tensor containing the orignal coordinates

    Returns:
        sparse_tensor (ME.SparseTensor):
            Tensor containing the sorted coordinates
    """
    # Sort the coordinates
    weights = torch.tensor([1e15, 1e10, 1e5, 1], dtype=torch.int64, device=sparse_tensor.device) 
    sortable_vals = (sparse_tensor.C * weights).sum(dim=1)
    sorted_coords_indices = sortable_vals.argsort()

    sparse_tensor = ME.SparseTensor(
        features=sparse_tensor.F[sorted_coords_indices],
        coordinates=sparse_tensor.C[sorted_coords_indices],
        tensor_stride=sparse_tensor.tensor_stride,
        device=sparse_tensor.device
    )
    return sparse_tensor



def sort_points(points):
    """
    Sort the coordinates of torch list sized Nx4

    Parameters:
        points (torch.tensor):
            Tensor containing the orignal coordinates Nx4

    Returns:
        points (torch.tensor):
            Tensor containing the orignal coordinates Nx4
    """
    # Sort the coordinates
    weights = torch.tensor([1e15, 1e10, 1e5, 1], dtype=torch.int64, device=points.device) 
    sortable_vals = (points * weights).sum(dim=1)
    sorted_coords_indices = sortable_vals.argsort()

    points = points[sorted_coords_indices]
    return points

def pc_metrics(reference, distorted, metric_path, data_path, resolution):
    """
    Compute pointcloud metrics using the mpeg pcc metrics implementation
    if available, else use our own implementation 

    Parameters:
        reference (o3d.geometry.PointCloud | string)
            Reference Point Cloud or path to it
        distorted (o3d.geometry.PointCloud):
            Distorted Point Cloud
        metric_path (str):
            Path to mpeg-pcc-metric binary
        resolution (int):
            Voxel resolution for PSNR normalization

    Returns:
    metrics (dict):
        Dictionary containing the results of the metric computation
    """
    # Folder initialization
    data_path = os.path.join(data_path, "tmp")
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    # Reference to disk
    if isinstance(reference, o3d.geometry.PointCloud):
        ref_path = os.path.join(data_path, "ref.ply") 
        save_ply(ref_path, reference)
    else:
        ref_path = os.path.join(reference)

    # Distorted to disk
    if isinstance(distorted, o3d.geometry.PointCloud):
        distorted_path = os.path.join(data_path, "distorted.ply") 
        save_ply(distorted_path, distorted, has_normals=True)
    else:
        distorted_path = os.path.join(distorted)

    # Calling external mpeg-pcc-metric
    command = [metric_path,
               '--fileA={}'.format(ref_path),
               '--fileB={}'.format(distorted_path),
               '--resolution={}'.format(resolution),
               '--color=1'
                ]
    result = subprocess.run(command, stdout=subprocess.PIPE)

    # Parse outputs to metric dict
    string = result.stdout
    lines = string.decode().split('\n')
    start = 0
    for i, line in enumerate(lines):
        if "infile1 (A)" in line:
            start = i
            break

    prefix = ["AB_", "BA_", "sym_"]
    metrics = {}
    for j in range(3):
        metrics[prefix[j] + "p2p_mse"] = float(lines[start+1].split(':')[-1].strip())
        metrics[prefix[j] + "p2p_psnr"] = float(lines[start+2].split(':')[-1].strip())

        metrics[prefix[j] + "d2_mse"] = float(lines[start+3].split(':')[-1].strip())
        metrics[prefix[j] + "d2_psnr"] = float(lines[start+4].split(':')[-1].strip())
        start += 2

        metrics[prefix[j] + "y_mse"] = float(lines[start+3].split(':')[-1].strip())
        metrics[prefix[j] + "u_mse"] = float(lines[start+4].split(':')[-1].strip())
        metrics[prefix[j] + "v_mse"] = float(lines[start+5].split(':')[-1].strip())
        metrics[prefix[j] + "y_psnr"] = float(lines[start+6].split(':')[-1].strip())
        metrics[prefix[j] + "u_psnr"] = float(lines[start+7].split(':')[-1].strip())
        metrics[prefix[j] + "v_psnr"] = float(lines[start+8].split(':')[-1].strip())

        # Compute YUV (using weights 0.75, 0.125, 0.125)
        metrics[prefix[j] + "yuv_psnr"] = (1/8) * (6 * metrics[prefix[j] + "y_psnr"] + metrics[prefix[j] + "u_psnr"] + metrics[prefix[j] + "v_psnr"])
        metrics[prefix[j] + "yuv_mse"] = (1/8) * (6 * metrics[prefix[j] + "y_mse"] + metrics[prefix[j] + "u_mse"] + metrics[prefix[j] + "v_mse"])

        start+=9

    return metrics

def pcqm(reference, distorted, pcqm_path, data_path, settings=None):
    """
    Compute PCQM with 

    Parameters:
        reference (o3d.geometry.PointCloud | string):
            Reference Point Cloud or path to it
        distorted (o3d.geometry.PointCloud):
            Distorted Point Cloud
        pcqm_path (str):
            Path to the PCQM binary
        settings (dictionary, default=None):
            Extra Settings for PCQM

    Returns:
        pcqm: float
            PCQM value
    """
    # Initialization
    cwd = os.getcwd()
    pcqm_path = os.path.join(cwd, pcqm_path)
    os.chdir(pcqm_path)
    data_path = os.path.join(cwd, data_path, "tmp")
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    # Save reference to disk
    if isinstance(reference, o3d.geometry.PointCloud):
        ref_path = os.path.join(data_path, "ref.ply") 
        save_ply(ref_path, reference)
    else:
        ref_path = os.path.join(cwd, reference)

    # Save distorted to disk
    if isinstance(distorted, o3d.geometry.PointCloud):
        distorted_path = os.path.join(data_path, "distorted.ply") 
        save_ply(distorted_path, distorted)
    else:
        distorted_path = os.path.join(cwd, distorted)

    # Call PCQM
    command = [pcqm_path + "/PCQM", ref_path, distorted_path, "-fq", "-r 0.004", "-knn 20", "-rx 2.0"]
    result = subprocess.run(command, stdout=subprocess.PIPE)

    # Parse output to pcqm result
    string = result.stdout
    lines = string.decode().split('\n')
    penultimate_line = lines[-3]
    pcqm_value_str = penultimate_line.split(':')[-1].strip() 
    pcqm_value = float(pcqm_value_str)  

    os.chdir(cwd)
    return pcqm_value


def save_ply(path, ply, has_normals=False):
    """
    Save a point cloud to a ply file. Exchange the o3d header through our header for 
    usage in the metric dependencies.
    """
    o3d.io.write_point_cloud(path, ply, write_ascii=True)

    with open(path, "r") as ply_file:
        lines = ply_file.readlines()

    # Extract the header
    header = []
    data_lines = []
    header_found = False
    
    for line in lines:
        header.append(line)
        if line.strip() == "end_header":
            header_found = True
        elif header_found:
            data_lines.append(line)

    # Update the property data type from double to float in the header
    new_header = []
    for line in header:
        if "property double" in line:
            new_header.append(line.replace("double", "float"))
        else:
            new_header.append(line)

    # Convert the data values from double to float
    data = np.genfromtxt(data_lines, dtype=np.float64)

    if has_normals:
        structured_data = np.zeros(data.shape[0], dtype=[('int1', 'i4'), ('int2', 'i4'), ('int3', 'i4'),
                                                        ('normal_x', 'f4'), ('normal_y', 'f4'), ('normal_z', 'f4'),
                                                        ('int4', 'i4'), ('int5', 'i4'), ('int6', 'i4')])
        # Assuming the data has normals, adjust indices accordingly
        structured_data['int1'] = data[:, 0].astype(np.int32)
        structured_data['int2'] = data[:, 1].astype(np.int32)
        structured_data['int3'] = data[:, 2].astype(np.int32)
        structured_data['normal_x'] = data[:, 3].astype(np.float32)
        structured_data['normal_y'] = data[:, 4].astype(np.float32)
        structured_data['normal_z'] = data[:, 5].astype(np.float32)
        structured_data['int4'] = data[:, 6].astype(np.int32)
        structured_data['int5'] = data[:, 7].astype(np.int32)
        structured_data['int6'] = data[:, 8].astype(np.int32)
    else:
        structured_data = np.zeros(data.shape[0], dtype=[('int1', 'i4'), ('int2', 'i4'), ('int3', 'i4'),
                                                        ('int4', 'i4'), ('int5', 'i4'), ('int6', 'i4')])
        # No normals, map accordingly
        structured_data['int1'] = data[:, 0].astype(np.int32)
        structured_data['int2'] = data[:, 1].astype(np.int32)
        structured_data['int3'] = data[:, 2].astype(np.int32)
        structured_data['int4'] = data[:, 3].astype(np.int32)
        structured_data['int5'] = data[:, 4].astype(np.int32)
        structured_data['int6'] = data[:, 5].astype(np.int32)

    # Save the modified PLY file
    with open(path, "w") as ply_file:
        for line in new_header:
            ply_file.write(line)

        for row in structured_data:
            ply_file.write(" ".join(map(str, row)) + "\n")
        

def remove_gpcc_header(path, gpcc=True):
    """
    Remove the header introduced by G-PCC decompression for o3d parsing (color-labels are permuted).
    """
    with open(path, "r") as ply_file:
        lines = ply_file.readlines()

    header = []
    for i, line in enumerate(lines):
        header.append(line)
        if line.strip() == "end_header":
            break

    # Update the property data type from double to float
    new_header = []
    for line in header:
        if "face" in line or "list" in line:
            continue
        if gpcc:
            if "green" in line:
                new_header.append(line.replace("green", "red"))
            elif "blue" in line:
                new_header.append(line.replace("blue", "green"))
            elif "red" in line:
                new_header.append(line.replace("red", "blue"))
            else:
                new_header.append(line)
        else:
            new_header.append(line)

    # Convert the data values from double to float
    data_lines = lines[i + 1:]
    
    # Save the modified PLY file
    with open(path, "w") as ply_file:
        for line in new_header:
            ply_file.write(line)
        for row in data_lines:
            ply_file.write(row)



def compress_model_ours(experiment, model, data, q_a, q_g, block_size, device, base_path):
    """
    Compress a point cloud using our model
    """
    points = data["src"]["points"].to(device, dtype=torch.float)
    colors = data["src"]["colors"].to(device, dtype=torch.float)
    source = torch.concat([points, colors], dim=2)[0]
    N = source.shape[0]

    # Bin path
    bin_path = os.path.join(base_path,
                            experiment,
                            "tmp")
    if not os.path.exists(bin_path): 
        os.mkdir(bin_path)
    bin_path = os.path.join(bin_path, "bitstream.bin")

    q = torch.tensor([[q_g, q_a]], device=device, dtype=torch.float)

    # Compression
    torch.cuda.synchronize()
    t0 = time.time()
    model.compress(source, q, block_size=block_size, path=bin_path)
    torch.cuda.synchronize()
    t_compress = time.time() - t0

    # Run decompression
    torch.cuda.synchronize()
    t0 = time.time()
    reconstruction = model.decompress(path=bin_path)
    torch.cuda.synchronize()
    t_decompress = time.time() - t0
                    
    # Rebuild point clouds
    source_pc = get_o3d_pointcloud(source)
    rec_pc = get_o3d_pointcloud(reconstruction)

    bpp = os.path.getsize(bin_path) * 8 / N

    return source_pc, rec_pc, bpp, t_compress, t_decompress



def compress_related(experiment, data, q_a, q_g, base_path):
    """
    Compress a point cloud using V-PCC/G-PCC
    """
    path = os.path.join(base_path,
                        experiment,
                        "tmp")
    if not os.path.exists(path):
        os.mkdir(path)

    # Directories
    src_dir = os.path.join(path, "points_enc.ply")
    rec_dir = os.path.join(path, "points_dec.ply")
    bin_dir = os.path.join(path, "points_enc.bin")

    N = data["src"]["points"].shape[1]

    # Data processing
    dtype = o3d.core.float32
    c_dtype = o3d.core.uint8
    points = data["src"]["points"].to(dtype=torch.float)
    colors = torch.clamp(data["src"]["colors"].to(dtype=torch.float) * 255, 0, 255)
    p_tensor = o3d.core.Tensor(points.detach().cpu().numpy()[0, :, :], dtype=dtype)
    p_colors = o3d.core.Tensor(colors.detach().cpu().numpy()[0, :, :], dtype=c_dtype)
    source = o3d.t.geometry.PointCloud(p_tensor)
    source.point.colors = p_colors
    o3d.t.io.write_point_cloud(src_dir, source, write_ascii=True)

    if experiment == "G-PCC":
        # Compress the point cloud using G-PCC
        command = ['./dependencies/mpeg-pcc-tmc13/build/tmc3/tmc3',
                '--mode=0',
                '--trisoupNodeSizeLog2=0',
                '--mergeDuplicatedPoints=1',
                '--neighbourAvailBoundaryLog2=8',
                '--intra_pred_max_node_size_log2=6',
                '--positionQuantizationScale={}'.format(q_g),
                '--maxNumQtBtBeforeOt=4',
                '--minQtbtSizeLog2=0',
                '--planarEnabled=1',
                '--planarModeIdcmUse=0',
                '--convertPlyColourspace=1',

                '--transformType=0',
                '--qp={}'.format(q_a),
                '--qpChromaOffset=-2',
                '--bitdepth=8',
                '--attrOffset=0',
                '--attrScale=1',
                '--attribute=color',

                '--uncompressedDataPath={}'.format(src_dir),
                '--compressedStreamPath={}'.format(bin_dir)]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        out = result.stdout.decode()
        output_lines = out.split("\n")
        processing_time_line = None
        for line in output_lines:
            if "Processing time (user)" in line:
                processing_time_line = line
        t_compress = float(processing_time_line.split()[-2])

        bpp = os.path.getsize(bin_dir) * 8 / N

        # Decode
        command = ['./dependencies/mpeg-pcc-tmc13/build/tmc3/tmc3',
                '--mode=1',
                '--convertPlyColourspace=1',
                '--outputBinaryPly=0',
                '--reconstructedDataPath={}'.format(rec_dir),
                '--compressedStreamPath={}'.format(bin_dir)]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        out = result.stdout.decode()
        output_lines = out.split("\n")
        processing_time_line = None
        for line in output_lines:
            if "Processing time (user)" in line:
                processing_time_line = line
        t_decompress = float(processing_time_line.split()[-2])

        # Read ply (o3d struggles with GBR order)
        remove_gpcc_header(rec_dir, gpcc=True)
        rec_pc = o3d.io.read_point_cloud(rec_dir)
        colors = np.asarray(rec_pc.colors)
        colors = colors[:, [2,0,1]]
        rec_pc.colors=o3d.utility.Vector3dVector(colors)

        # Clean up
        os.remove(rec_dir)
        os.remove(src_dir)
        os.remove(bin_dir)

    elif experiment == "V-PCC": 
        # TODO: Not complete, might want to rebuild at some point
        occPrecision = 4 if q_g > 16 else 2
        command = ['./dependencies/mpeg-pcc-tmc2/bin/PccAppEncoder',
                '--configurationFolder=./dependencies/mpeg-pcc-tmc2/cfg/',
                '--config=./dependencies/mpeg-pcc-tmc2/cfg/common/ctc-common.cfg',
                '--config=./dependencies/mpeg-pcc-tmc2/cfg/condition/ctc-all-intra.cfg',
                '--config=./dependencies/mpeg-pcc-tmc2/cfg/sequence/{}_vox10.cfg'.format(sequence), # Overwrite per sequence later
                '--frameCount=1',
                '--geometryQP={}'.format(q_g),
                '--attributeQP={}'.format(q_a),
                '--occupancyPrecision={}'.format(occPrecision),
                '--compressedStreamPath={}'.format(bin_dir),
                '--uncompressedDataPath={}'.format(src_dir)]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        out = result.stdout.decode()
        output_lines = out.split("\n")
        processing_time_line = None
        for line in output_lines:
            if "Processing time (user.self)" in line:
                processing_time_line = line
        t_compress = float(processing_time_line.split()[-2])

        bpp = os.path.getsize(bin_dir) * 8 / N

        # Decode
        command = ['./dependencies/mpeg-pcc-tmc2/bin/PccAppDecoder',
                '--inverseColorSpaceConversionConfig=./dependencies/mpeg-pcc-tmc2/cfg/hdrconvert/yuv420torgb444.cfg',
                '--reconstructedDataPath={}'.format(rec_dir),
                '--compressedStreamPath={}'.format(bin_dir)]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        out = result.stdout.decode()
        output_lines = out.split("\n")
        processing_time_line = None
        for line in output_lines:
            if "Processing time (user.self)" in line:
                processing_time_line = line
        t_decompress = float(processing_time_line.split()[-2])

        # Read ply (o3d struggles with GBR order)
        remove_gpcc_header(rec_dir, gpcc=False)
        rec_pc = o3d.io.read_point_cloud(rec_dir)
        colors = np.asarray(rec_pc.colors)
        rec_pc.colors=o3d.utility.Vector3dVector(colors)

    elif experiment=="IT-DL-PCC":
        command = ['python3', './dependencies/IT-DL-PCC/src/IT-DL-PCC.py',
            '--with_color',
            '--cuda', 
            'compress',
            '{}'.format(src_dir),
            './dependencies/IT-DL-PCC/models/Joint/Codec/{}/checkpoint_best_loss.pth.tar'.format(q_g),
            '{}'.format(path),
            '--scale=1',
            '--use_fast_topk',
            '--blk_size=256',
        ]

        t0 = time.time()
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        t_compress = time.time() - t0

        out = result.stdout.decode()

        bin_dir = os.path.join(path, "points_enc/points_enc.gz")
        bpp = os.path.getsize(bin_dir) * 8 / N

        command = ['python3', './dependencies/IT-DL-PCC/src/IT-DL-PCC.py',
            '--with_color',
            '--cuda',
            'decompress',
            '{}'.format(bin_dir),
            './dependencies/IT-DL-PCC/models/Joint/Codec/{}/checkpoint_best_loss.pth.tar'.format(q_g)
        ]

        t0 = time.time()
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        t_decompress = time.time() - t0

        out = result.stdout.decode()
        rec_dir = os.path.join(path, "points_enc/points_enc.gz.dec.ply")
        rec_pc = o3d.io.read_point_cloud(rec_dir)
        colors = np.asarray(rec_pc.colors)
        rec_pc.colors=o3d.utility.Vector3dVector(colors)

        #Cleanup
        os.remove(rec_dir)
        os.remove(src_dir)
        os.remove(bin_dir)


    # Reconstruct source
    points = data["src"]["points"]
    colors = data["src"]["colors"]
    source = torch.concat([points, colors], dim=2)[0]
    source_pc = get_o3d_pointcloud(source)
    return source_pc, rec_pc, bpp, t_compress, t_decompress


def overlapping_mask(tensor1, tensor2):
    """
    Get the overlapping coordinates of two tensors

    Parameters:
        tensor1 (torch.tensor):
            tensor 1 for which the mask is created
        tensor2 (torch.tensor):
            tensor 2 for checking where tensor 1 overlaps with
    
    Returns:
        mask (torch.tensor):
            Mask for tensor1 where it overlaps with tensor2
    """
    scaling_factors = torch.tensor([1, 1e2, 1e7, 1e12], dtype=torch.int64, device=tensor1.C.device)

    tensor1_flat = (tensor1.C.to(torch.int64) * scaling_factors).sum(dim=1)
    tensor2_flat = (tensor2.C.to(torch.int64) * scaling_factors).sum(dim=1)

    unique_tensor1, counts_tensor1 = torch.unique(tensor1_flat, return_counts=True)
    unique_tensor2, counts_tensor2 = torch.unique(tensor2_flat, return_counts=True)

    tensor1_duplicates = unique_tensor1[counts_tensor1 > 1]
    tensor2_duplicates = unique_tensor2[counts_tensor2 > 1]

    if tensor1_duplicates.numel() > 0:
        print("Duplicate values in tensor1_flat:", tensor1_duplicates)
    if tensor2_duplicates.numel() > 0:
        print("Duplicate values in tensor2_flat:", tensor2_duplicates)

    mask = torch.isin(tensor1_flat, tensor2_flat)
    return mask