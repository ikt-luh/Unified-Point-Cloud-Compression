import open3d as o3d

import open3d as o3d
import numpy as np

def compute_adaptive_normals(pcd, scale=5, max_nn=30, orient=True):
    # Estimate average nearest neighbor distance
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = scale * avg_dist

    print(f"Using adaptive radius: {radius:.4f}, max_nn: {max_nn}")

    # Estimate normals with hybrid search
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius, max_nn=max_nn))
    
    # Normalize normals
    pcd.normalize_normals()

    # Optional: orient consistently (tangent plane fitting)
    if orient:
        pcd.orient_normals_consistent_tangent_plane(k=30)

    return pcd


def process_pointcloud(file_path, output_path):
    pcd = o3d.io.read_point_cloud(file_path)
    if not pcd.has_points():
        raise ValueError(f"No points found in {file_path}")
    
    print(f"Estimating normals for: {file_path}")
    pcd = compute_adaptive_normals(pcd)
    save_ply(output_path, pcd, has_normals=True)
    print(f"Saved with normals to: {output_path}")

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
if __name__ == "__main__":
    input_files = ["../data/datasets/jpeg_testset/EPFL_vox13.ply", "../data/datasets/jpeg_testset/CITIUSP_vox13.ply"]
    output_files = ["../data/datasets/jpeg_testset/EPFL_vox13_n.ply", "../data/datasets/jpeg_testset/CITIUSP_vox13_n.ply"]

    for in_file, out_file in zip(input_files, output_files):
        process_pointcloud(in_file, out_file)