import os
import time
import yaml

import torch
import open3d as o3d
import pandas as pd
import numpy as np
import copy
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
import subprocess
from plyfile import PlyData

import utils
from model.model import ColorModel
from data.dataloader import StaticDataset
from metrics.metric import PointCloudMetric

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

norm = Normalize(vmin=0.0, vmax=10**(-2))
# Paths
base_path = "./results"
data_path = "./data/datasets/full_128" 
ref_paths = {
     #"thaidancer" : "./data/datasets/raw/jpeg_testset/Thaidancer_viewdep_vox12.ply",
     #"bouquet" : "./data/datasets/raw/jpeg_testset/RWT130Bouquet.ply",
     #"stmichael" : "./data/datasets/raw/jpeg_testset/RWT70StMichael.ply",
     #"soldier" : "./data/datasets/raw/jpeg_testset/soldier_vox10_0690.ply",

     #"boxer" : "./data/datasets/raw/jpeg_testset/boxer_viewdep_vox12.ply",
     "House" : "./data/datasets/raw/jpeg_testset/House_without_roof_00057_vox12.ply",
     "CITISUP" : "./data/datasets/raw/jpeg_testset/CITIUSP_vox13.ply",
     "Facade" : "./data/datasets/raw/jpeg_testset/Facade_00009_vox12.ply",

     "EPFL" : "./data/datasets/raw/jpeg_testset/EPFL_vox13.ply",
     "Arco" : "./data/datasets/raw/jpeg_testset/Arco_Valentino_Dense_vox12.ply",
     "shiva" : "./data/datasets/raw/jpeg_testset/Shiva_00035_vox12.ply",
     "Unicorn" : "./data/datasets/raw/jpeg_testset/ULB_Unicorn_vox13_n.ply",
     }
resolutions ={
     "soldier" : 1023, 
     "boxer" : 4095,
     "thaidancer" : 4095,
     "bouquet" : 1023,

     "stmichael" : 1023,
     "CITISUP" : 8191,
     "EPFL" : 8191,
     "Facade" : 4095,

     "House" : 4095,
     "shiva" : 4095,
     "Unicorn" : 8191,
     "Arco" : 4095,
}
block_sizes ={
     "soldier" : 1024, 
     "boxer" : 512,
     "thaidancer" : 512,
     "bouquet" : 512,

     "stmichael" : 1024,
     "CITISUP" : 512,
     "EPFL" : 512,
     "Facade" : 512,

     "House" : 512,
     "shiva" : 1024,
     "Unicorn" : 1024,
     "Arco" : 1024,
}


device_id = 3
experiments = [
    "Ours",
    #"V-PCC",
    #"G-PCC",
    ]

related_work = [
    "G-PCC",
    "V-PCC"
]

def run_testset(experiments):
    # Device
    device = torch.device(device_id)
    torch.cuda.set_device(device)

    torch.no_grad()
        
    # Dataloader
    test_set = StaticDataset(data_path, split="test", transform=None, partition=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    for experiment in experiments:
        experiment_results = []

        # Set model and QPs
        if experiment not in related_work:
            #q_as = np.arange(6) * 0.2
            #q_gs = np.arange(6) * 0.2
            q_as = np.arange(21) * 0.05
            q_gs = np.arange(21) * 0.05
            q_as = [0.5]
            q_gs = [0.5]

            weight_path = os.path.join(base_path, experiment, "weights.pt")
            config_path = os.path.join(base_path, experiment, "config.yaml")

            with open(config_path, "r") as config_file:
                config = yaml.safe_load(config_file)

            model = ColorModel(config["model"])
            model.load_state_dict(torch.load(weight_path))
            model.to(device)
            model.eval()
            model.update()
        elif experiment == "G-PCC":
            q_as = np.arange(21, 52)
            q_gs = [0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 0.875, 0.9375]
        elif experiment == "V-PCC":
            #q_as = [42, 37, 32, 27, 22]
            q_as = np.arange(22, 43)
            q_gs = [32, 28, 24, 20, 16]
        

        with torch.no_grad():
            for s, sequence in enumerate(ref_paths.keys()):
                ref_path = ref_paths[sequence]
                """
                with open(ref_path, "rb") as f:
                    plydata = PlyData.read(f)

                # Load ply and convert to points array
                points = np.vstack([plydata['vertex'].data['x'], 
                                    plydata['vertex'].data['y'], 
                                    plydata['vertex'].data['z']]).T
                colors = np.vstack([plydata['vertex'].data['red'], 
                                    plydata['vertex'].data['green'], 
                                    plydata['vertex'].data['blue']]).T / 255.0
                #cd = o3d.io.read_point_cloud(ref_path, format="ply")
                points = torch.from_numpy(points).unsqueeze(0).float()
                colors = torch.from_numpy(colors).unsqueeze(0).float()
                data = { "src" : {"points" : points, "colors": colors}}
                """

                pcd = o3d.io.read_point_cloud(ref_path)

                # Step 2: Convert Open3D point cloud data to NumPy arrays
                points = np.asarray(pcd.points)  # Extract points (N, 3)
                colors = np.asarray(pcd.colors)  # Extract colors (N, 3)

                # Step 3: Convert NumPy arrays to PyTorch tensors
                points = torch.from_numpy(points).unsqueeze(0).float()  # Shape (1, N, 3)
                colors = torch.from_numpy(colors).unsqueeze(0).float()  # Shape (1, N, 3)

                # Step 4: Prepare the data in the desired format
                data = {"src": {"points": points, "colors": colors}}
                for j, q_g in enumerate(q_gs):
                    for i, q_a in enumerate(q_as):
                        # Get info
                        t0 = time.time()

                        block_size = block_sizes[sequence]


                        # Run model
                        if experiment not in related_work:
                            source_pc, rec_pc, bpp, t_compress, t_decompress = utils.compress_model_ours(experiment,
                                                                                             model,
                                                                                             data,
                                                                                             q_a, 
                                                                                             q_g, 
                                                                                             block_size,
                                                                                             device,
                                                                                             base_path)
                        else:
                            source_pc, rec_pc, bpp, t_compress, t_decompress = utils.compress_related(experiment,
                                                                                                data,
                                                                                                q_a,
                                                                                                q_g,
                                                                                                base_path)

                        tmp_path = os.path.join(base_path,
                                                experiment)
                        """
                        results = utils.pc_metrics(ref_path, 
                                                     rec_pc, 
                                                     "dependencies/mpeg-pcc-tmc2/bin/PccAppMetrics",
                                                     tmp_path,
                                                     resolution=resolutions[sequence])
                        """
                        results = {}
                        results["pcqm"] = utils.pcqm(ref_path, 
                                                     rec_pc, 
                                                     "dependencies/PCQM/build",
                                                     tmp_path)

                        # Save results
                        results["bpp"] = bpp
                        results["sequence"] = sequence
                        results["frameIdx"] = 0 #TODO
                        results["t_compress"] = t_compress
                        results["t_decompress"] = t_decompress
                        results["q_a"] = q_a
                        results["q_g"] = q_g
                        experiment_results.append(results)

                        torch.cuda.empty_cache()
                        t1 = time.time() - t0
                        total = len(test_loader) * len(q_as) * len(q_gs)
                        done = (s * len(q_as) * len(q_gs)) + (i * len(q_gs)) + j + 1
                        print("[{}/{}] Experiment: {} | Sequence: {} @ q_a:{:.2f} q_g:{:.2f} | {:2f}s | PCQM:{:4f} bpp:{:2f} t_comp:{:2f}s ".format(done,
                                                                                                                                     total,
                                                                                                                                     experiment,
                                                                                                                               sequence, 
                                                                                                                               q_a, 
                                                                                                                               q_g,  
                                                                                                                               t1,
                                                                                                                               results["pcqm"],
                                                                                                                               results["bpp"],
                                                                                                                               t_compress + t_decompress))
                        # Renders of the pointcloud
                        point_size = 1.0 if sequence in ["longdress", "soldier", "loot", "longdress"] else 2.0
                        path = os.path.join(base_path,
                                            experiment, 
                                            "renders_test",
                                            "{}_a{}_g{}_{}.png".format(sequence, str(q_a), str(q_g), "{}"))
                        utils.render_pointcloud(rec_pc, path, point_size=point_size)

                        # Renders of the original
                        path = os.path.join(base_path,
                                            experiment, 
                                            "renders_test",
                                            "{}_original_{}.png".format(sequence, "{}"))
                        utils.render_pointcloud(source_pc, path, point_size=point_size)
                        """
                        fig, ax = plt.subplots(figsize=(2, 4), layout='constrained')
                        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap="magma"),
                                cax=ax, orientation='vertical', label='MSE')
                        for t in cbar.ax.get_yticklabels():
                            t.set_fontsize(18)
                        cbar.ax.set_ylabel("MSE", fontsize=18)

                        fig.savefig(os.path.join(base_path, experiment, "renders_test", "error_bar.png"), bbox_inches="tight")
                        exit(0)
                        # Calculate the median x-coordinate
                        median_x = np.median(np.asarray(error_point_cloud.points)[:, 0])

                        # Split error_point_cloud
                        error_points = np.asarray(error_point_cloud.points)
                        left_error_indices = np.where(error_points[:, 0] <= median_x)[1]
                        right_error_indices = np.where(error_points[:, 0] > median_x)[1]

                        left_error_cloud = error_point_cloud.select_by_index(left_error_indices)
                        right_error_cloud = error_point_cloud.select_by_index(right_error_indices)

                        # Split rec_pc
                        rec_pc_points = np.asarray(rec_pc.points)
                        left_rec_pc_indices = np.where(rec_pc_points[:, 0] <= median_x)[0]
                        right_rec_pc_indices = np.where(rec_pc_points[:, 0] > median_x)[0]

                        left_rec_pc = rec_pc.select_by_index(left_rec_pc_indices)
                        right_rec_pc = rec_pc.select_by_index(right_rec_pc_indices)

                        split_cloud = o3d.geometry.PointCloud()
                        split_cloud.points = o3d.utility.Vector3dVector(np.concatenate([left_error_cloud.points, right_rec_pc.points], axis=0))
                        split_cloud.colors = o3d.utility.Vector3dVector(np.concatenate([left_error_cloud.colors, right_rec_pc.colors], axis=0))
                        path = os.path.join(base_path,
                                            experiment, 
                                            "renders_test", 
                                            "split_y_{}_{}_{}.png".format(sequence, str(i), "{}"))
                        utils.render_pointcloud(split_cloud, path)

                        # Ply
                        path = os.path.join(base_path,
                                            experiment, 
                                            "plys", 
                                            "{}_{:04d}_rec{}.ply".format(sequence, results["frameIdx"], str(i)))
                        """

                    # Save the results as .csv
                    df = pd.DataFrame(experiment_results)
                    results_path = os.path.join(base_path, experiment, "test.csv")
                    df.to_csv(results_path)





if __name__ == "__main__":
    run_testset(experiments)
