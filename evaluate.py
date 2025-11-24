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
from datetime import datetime
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
     "thaidancer" : "./data/datasets/jpeg_testset/Thaidancer_viewdep_vox12.ply",
     "bouquet" : "./data/datasets/jpeg_testset/RWT130Bouquet.ply",
     "stmichael" : "./data/datasets/jpeg_testset/RWT70StMichael.ply",
     "boxer" : "./data/datasets/jpeg_testset/boxer_viewdep_vox12.ply",
     "House" : "./data/datasets/jpeg_testset/House_without_roof_00057_vox12.ply",
     "Facade" : "./data/datasets/jpeg_testset/Facade_00009_vox12.ply",
     "Arco" : "./data/datasets/jpeg_testset/Arco_Valentino_Dense_vox12.ply",
     "shiva" : "./data/datasets/jpeg_testset/Shiva_00035_vox12.ply",
     "Unicorn" : "./data/datasets/jpeg_testset/ULB_Unicorn_vox13_n.ply",
     "CITISUP" : "./data/datasets/jpeg_testset/CITIUSP_vox13_n.ply",
     "EPFL" : "./data/datasets/jpeg_testset/EPFL_vox13_n.ply",
     }
resolutions ={
     "longdress" : 1023, "soldier" : 1023, "loot" : 1023, "redandblack" : 1023, 
     "boxer" : 4095, "thaidancer" : 4095, "bouquet" : 1023, "stmichael" : 1023,
     "CITISUP" : 8191, "EPFL" : 8191, "Facade" : 4095, "House" : 4095,
     "shiva" : 4095, "Unicorn" : 8191, "Arco" : 4095,
}
block_sizes ={
     "soldier" : 1024, "longdress" : 1024, "loot" : 1024, "redandblack" : 1024, 
     "boxer": 1024, "thaidancer" : 1024, "bouquet" : 1024, "stmichael" : 1024,
     "CITISUP" : 1024, "EPFL" : 1024, "Facade" : 1024, "House" : 1024,
     "shiva" : 1024, "Unicorn" : 1024, "Arco" : 1024,
}


device_id = 3
experiments = [
    "Main",
    #"Ablation_fixed_R1",
    #"Ablation_fixed_R2",
    #"Ablation_fixed_R3",
    #"Ablation_fixed_R4",
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
        date_str = datetime.now().strftime("%Y-%m-%d")
        results_path = os.path.join(base_path, experiment, f"test_{date_str}.csv")
        experiment_results = []

        # Set model and QPs
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

            for scaling_factor in [1, 2, 4]:
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
                                                                                                scaling_factor,
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
                                            "renders_test/{}".format(sequence),
                                            "{}_s{}_a{}_g{}_{}.png".format(sequence, str(scaling_factor), str(q_a), str(q_g), "{}"))
                        utils.render_pointcloud(rec_pc, path, point_size=point_size)

                        # Renders of the original
                        path = os.path.join(base_path,
                                            experiment, 
                                            "renders_test/{}".format(sequence),
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
                        results["scale_factor"] = scaling_factor
                        experiment_results.append(results)

                        torch.cuda.empty_cache()
                        t1 = time.time() - t0
                        total = len(ref_paths.keys()) * len(q_as) * len(q_gs) * 3
                        done = (scaling_factor * len(q_as) * len(q_gs)) + (j * len(q_gs)) + i + 1
                        print(f"[{done}/{total}] Experiment: {experiment} | Sequence: {sequence} "
                                f"@ q_a:{q_a:.2f} q_g:{q_g:.2f} | {t1:.2f}s | "
                                f"PCQM:{results['pcqm']:.4f} bpp:{results['bpp']:.2f} "
                                f"t_comp:{(t_compress + t_decompress):.2f}s")

                    # Save the results as .csv
                    df = pd.DataFrame(experiment_results)
                    df.to_csv(results_path)

if __name__ == "__main__":
    run_testset(experiments)