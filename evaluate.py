import os
import time
import yaml
import copy
import torch
import subprocess
import numpy as np
import open3d as o3d
import pandas as pd
import MinkowskiEngine as ME
import matplotlib.pyplot as plt
from plyfile import PlyData
from matplotlib.cm import ScalarMappable
from torch.utils.data import DataLoader
from matplotlib.colors import Normalize

import utils
from model.model import UnifiedModel
from metrics.metric import PointCloudMetric
from data.dataloader import StaticDataset

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)

# Paths
base_path = "./results"

ref_paths = {
     "loot" : "./data/datasets/8iVFB/loot_vox10_1200.ply",
     "longdress" : "./data/datasets/8iVFB/longdress_vox10_1300.ply",
     "soldier" : "./data/datasets/8iVFB/soldier_vox10_0690.ply",
     "redandblack" : "./data/datasets/8iVFB/redandblack_vox10_1550.ply",
     "basketball_player" : "./data/datasets/Owlii/basketball_player_vox11_00000200.ply",
     "dancer" : "./data/datasets/Owlii/dancer_vox11_00000001.ply",
     "exercise" : "./data/datasets/Owlii/exercise_vox11_00000001.ply",
     "model" : "./data/datasets/Owlii/model_vox11_00000001.ply",
     }
resolutions ={
     "longdress" : 1023, "soldier" : 1023, "loot" : 1023, "redandblack" : 1023, 
     "basketball_player" : 2047, "dancer" : 2047, "model" : 2047, "exercise" : 2047, 
}
block_sizes ={
     "soldier" : 1024, "longdress" : 1024, "loot" : 1024, "redandblack" : 1024, 
     "model" : 512, "exercise" : 512, "dancer" : 512, "basketball_player" : 512, 
}


device_id = 0
experiments = [
    #"G-PCC",
    #"IT-DL-PCC",
    #"CVPR_inverse_scaling_shepard"
    #"CVPR_inverse_scaling_fixed_R1",
    #"CVPR_inverse_scaling_fixed_R2",
    #"CVPR_inverse_scaling_fixed_R3",
    "CVPR_inverse_scaling",
    ]

related_work = [
    "G-PCC",
    "IT-DL-PCC",
]

def run_testset(experiments):
    # Device
    device = torch.device(device_id)
    torch.cuda.set_device(device)
    torch.autograd.set_grad_enabled(False)

    for experiment in experiments:
        experiment_results = []

        # Set model and QPs
        if experiment not in related_work:
            q_as = np.arange(11) * 0.1
            q_gs = np.arange(11) * 0.1

            weight_path = os.path.join(base_path, experiment, "weights.pt")
            config_path = os.path.join(base_path, experiment, "config.yaml")

            with open(config_path, "r") as config_file:
                config = yaml.safe_load(config_file)

            model = UnifiedModel(config["model"])
            model.load_state_dict(torch.load(weight_path))
            model.to(device)
            model.eval()
            model.update()
        elif experiment == "G-PCC":
            q_as = [51, 46, 40, 34, 28, 22]
            q_gs = [0.0625, 0.125, 0.25, 0.5, 0.75, 0.875, 0.9375]
        elif experiment == "IT-DL-PCC":
            q_as = [0]
            q_gs = [0.001, 0.002, 0.004, 0.0005, 0.00025, 0.000125]
        
        if os.path.exists("./dependencies/mpeg-pcc-dmetric-master/test/pc_error"):
            use_mpeg_metrics = True
        else:
            use_mpeg_metrics = False

        for s, sequence in enumerate(ref_paths.keys()):
            ref_path = ref_paths[sequence]

            pcd = o3d.io.read_point_cloud(ref_path)
            points = np.asarray(pcd.points) 
            colors = np.asarray(pcd.colors) 
            points = torch.from_numpy(points).unsqueeze(0).float()
            colors = torch.from_numpy(colors).unsqueeze(0).float()

            data = {"src": {"points": points, "colors": colors}}

            for j, q_g in enumerate(q_gs):
                for i, q_a in enumerate(q_as):
                    t0 = time.time()

                    block_size = block_sizes[sequence]

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

                    # Renders of the reconstruction
                    point_size = 0.1
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
                    tmp_path = os.path.join(base_path,
                                            experiment)

                    # Normal estimation for the reconstruction
                    rec_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=5.0))

                    # Metric computations
                    if use_mpeg_metrics:
                        results = utils.pc_metrics(ref_path, 
                                                        rec_pc, 
                                                        "dependencies/mpeg-pcc-dmetric-master/test/pc_error",
                                                        tmp_path,
                                                        resolution=resolutions[sequence])
                    else:
                        #source = utils.get_o3d_pointcloud(source_pc)
                        #rec = utils.get_o3d_pointcloud(rec_pc)
                        metric = PointCloudMetric(source_pc, rec_pc, resolution=resolutions[sequence], drop_duplicates=False)
                        results, error_vectors = metric.compute_pointcloud_metrics(drop_duplicates=True)

                    results["pcqm"] = utils.pcqm(ref_path, 
                                                    rec_pc, 
                                                    "dependencies/PCQM/build",
                                                    tmp_path)


                    # Save results
                    results["bpp"] = bpp
                    results["sequence"] = sequence
                    results["frameIdx"] = 0
                    results["t_compress"] = t_compress
                    results["t_decompress"] = t_decompress
                    results["q_a"] = q_a
                    results["q_g"] = q_g
                    experiment_results.append(results)

                    torch.cuda.empty_cache()
                    t1 = time.time() - t0
                    total = len(ref_paths.keys()) * len(q_as) * len(q_gs)
                    done = (s * len(q_as) * len(q_gs)) + (i * len(q_gs)) + j + 1
                    print(f"[{done}/{total}] Experiment: {experiment} | Sequence: {sequence} "
                            f"@ q_a:{q_a:.2f} q_g:{q_g:.2f} | {t1:.2f}s | "
                            f"PCQM:{results['pcqm']:.4f} bpp:{results['bpp']:.2f} "
                            f"t_comp:{(t_compress + t_decompress):.2f}s")

                # Save the results as .csv
                df = pd.DataFrame(experiment_results)
                results_path = os.path.join(base_path, experiment, "test.csv")
                df.to_csv(results_path)

if __name__ == "__main__":
    run_testset(experiments)