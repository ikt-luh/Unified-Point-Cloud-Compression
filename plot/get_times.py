import pandas as pd
import os

# Example settings format
settings = {
    "Main": [
        {"scale_factor": "1"},
        {"scale_factor": "2"},
        {"scale_factor": "4"}
    ],
    "IT-DL-PCC": [
        {"scale_factor": "1"},
    ]
}

stats = []

for method, filters in settings.items():
    path = f"../results/{method}/test.csv"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue

    df = pd.read_csv(path)

    for filt in filters:
        query_str = ' & '.join([f'`{k}` == {repr(v)}' for k, v in filt.items()])
        filtered = df.query(query_str)

        if filtered.empty:
            print(f"No matching rows for {method} with filter {filt}")
            continue

        stats.append({
            "method": method,
            "filter": filt,
            "t_comp_mean": filtered["t_comp"].mean(),
            "t_comp_std": filtered["t_comp"].std(),
            "t_comp_min": filtered["t_comp"].min(),
            "t_comp_max": filtered["t_comp"].max(),
            "t_decomp_mean": filtered["t_decomp"].mean(),
            "t_decomp_std": filtered["t_decomp"].std(),
            "t_decomp_min": filtered["t_decomp"].min(),
            "t_decomp_max": filtered["t_decomp"].max(),
        })

# Convert to DataFrame for display or further processing
results_df = pd.DataFrame(stats)
print(results_df)
