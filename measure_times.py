import os
import time
import yaml
import torch
import numpy as np
import pandas as pd
import open3d as o3d
from datetime import datetime

import utils
from model.model import UnifiedModel

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)

# Config
device_id = 0
base_path = "./results"
sequence = "soldier"
ref_path = f"./data/datasets/8iVFB/{sequence}_vox10_0690.ply"
N = 30  # Number of repetitions per setting

device = torch.device(device_id)
torch.cuda.set_device(device)
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
torch.autograd.set_grad_enabled(False)

# PC Loading
pcd = o3d.io.read_point_cloud(ref_path)
points = torch.from_numpy(np.asarray(pcd.points)).unsqueeze(0).float()
colors = torch.from_numpy(np.asarray(pcd.colors)).unsqueeze(0).float()
data = {"src": {"points": points, "colors": colors}}

#  Experiments
experiments = [
    "Main",
    "G-PCC",
    "IT-DL-PCC",
    "V-PCC"
]

#  Time measurements
date_str = datetime.now().strftime("%Y-%m-%d")
all_results = []

for experiment in experiments:
    exp_list = []
    if experiment == "Main":
        exp_list = [1, 2, 4]

        weight_path = os.path.join(base_path, experiment, "weights.pt")
        config_path = os.path.join(base_path, experiment, "config.yaml")

        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)

        model = UnifiedModel(config["model"])
        model.load_state_dict(torch.load(weight_path))
        model.to(device)
        model.eval()
        model.update()
    else:
        with open(f"results/{experiment}/settings.yaml", "r") as f:
                settings = yaml.safe_load(f)

        sequence_settings = settings["sequences"][sequence]

        if experiment == "G-PCC":
            enable_planar = sequence_settings["enable_planar"]
            for pQ, QP in zip(sequence_settings["pQs"], sequence_settings["QP"]):
                exp_list.append({"pQS": pQ, "QP": QP, "enable_planar": enable_planar})

        elif experiment == "V-PCC":
            seq_config = sequence_settings["seq_config"]
            for QA, QG, occPrec in zip(sequence_settings["GeometryQP"],
                                    sequence_settings["AttributeQP"],
                                    sequence_settings["occPrecision"]):
                exp_list.append({
                    "GeometryQP": QA,
                    "AttributeQP": QG,
                    "occPrecision": occPrec,
                    "seq_config": seq_config,
                    "sequence": sequence
                })

        elif experiment == "IT-DL-PCC":
            models = sequence_settings["model"]
            scales = sequence_settings["scales"]
            use_sr = sequence_settings["SR"]
            model = "0.001"
            for scale in scales:
                exp_list.append({"model": model, "scale": scale, "SR": False})
                if scale != 1 and 1 in use_sr:
                    exp_list.append({"model": model, "scale": scale, "SR": True})

    print(exp_list)
    for idx, sett in enumerate(exp_list):
        print(idx, sett)
        for rep in range(-1, N):
            torch.cuda.empty_cache()

            if experiment == "Main":
                source_pc, rec_pc, bpp, t_compress, t_decompress = utils.compress_model_ours(experiment,
                                                                                    model,
                                                                                    data,
                                                                                    1.0, 
                                                                                    1.0, 
                                                                                    sett,
                                                                                    1024,
                                                                                    device,
                                                                                    base_path)
            
            else:
                _, _, _, t_compress, t_decompress = utils.compress_related(
                    experiment, data, sett, base_path)

            if rep < 0:
                # Warump each setting
                continue
            result = {
                "experiment": experiment,
                "sequence": sequence,
                "rep": rep,
                "IDX": idx,
                "t_compress": t_compress,
                "t_decompress": t_decompress,
            }
            all_results.append(result)

            print(f"[{experiment}] idx:{idx} rep:{rep} "
                  f" t_comp:{t_compress:.2f}s t_decomp:{t_decompress:.2f}s")

# Save to CSV
df = pd.DataFrame(all_results)
out_path = f"time_measurements_{sequence}_{date_str}.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved timing results to {out_path}")
