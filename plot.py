import os
from metrics.bjontegaard import Bjontegaard_Delta, Bjontegaard_Model

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from scipy.interpolate import griddata
import scipy.stats as st
import pandas as pd
import numpy as np

from plot import style
from plot.style import runs, metric_labels

# Runs
path = "./results"
plots = "./plot/figures"
metrics = ["pcqm", "sym_y_psnr", "sym_p2p_psnr", "sym_d2_psnr", "sym_yuv_psnr"]

sota_comparison = ["CVPR_inverse_scaling", "G-PCC", "IT-DL-PCC", "DeepPCC", "YOGA"]
ablation_scaling = ["CVPR_inverse_scaling", "Final_L2_GDN_scale_rescale_ste_offsets_inverse_nn_vbr_btlnk"]
ablation_loss = ["CVPR_inverse_scaling", "CVPR_inverse_scaling_shepard"]
ablation_fixed = ["CVPR_inverse_scaling", "CVPR_inverse_scaling_fixed"]

datasets = {"8iVFBv2": ["soldier", "longdress", "loot", "redandblack", "8iVFBv2"],
            "Owlii": ["exercise", "model", "dancer", "basketball_player", "Owlii"]}

def plot_experiments():
    """
    Level 0 : Plot all results
    """
    # SotA Comparison
    data = load_csvs(sota_comparison)
    plot_rd_figs_all(data, "sota_comparison")
    
    # TODO
    compute_bd_deltas(data, sota_comparison, "CVPR_inverse_scaling", "sota_comparison")
    
    # Timing
    # Plot per run results
    pareto_data = {}
    for key, dataframe in data.items():
        pareto_df = plot_per_run_results(dataframe, key)
        pareto_data[key] = pareto_df

    # Scaling Comparison
    data = load_csvs(ablation_scaling)
    plot_rd_figs_all(data, "ablation_scaling")
    compute_bd_deltas(data, ablation_scaling, "CVPR_inverse_scaling", "ablation_scaling")
    pareto_data = {}
    for key, dataframe in data.items():
        pareto_df = plot_per_run_results(dataframe, key)
        pareto_data[key] = pareto_df

    # Loss Comparison
    data = load_csvs(ablation_loss)
    plot_rd_figs_all(data, "ablation_loss")
    compute_bd_deltas(data, ablation_loss, "CVPR_inverse_scaling", "ablation_loss")
    pareto_data = {}
    for key, dataframe in data.items():
        pareto_df = plot_per_run_results(dataframe, key)
        pareto_data[key] = pareto_df

    # Fixed Comparison
    data = load_csvs(ablation_fixed)
    plot_rd_figs_all(data, "ablation_fixed")
    compute_bd_deltas(data, ablation_fixed, "CVPR_inverse_scaling", "ablation_fixed")
    pareto_data = {}
    for key, dataframe in data.items():
        pareto_df = plot_per_run_results(dataframe, key)
        pareto_data[key] = pareto_df


def plot_per_run_results(dataframe, key):
    """
    Level 1 : Plot per run results
    """
    
    # Generate the path
    directory = os.path.join(plots, key)
    if not os.path.exists(directory):
        os.mkdir(directory)

    # Filter df for pareto fron
    pareto_df = get_pareto_df(dataframe)

    #plot_pareto_figs_single(pareto_df, key)
    plot_settings(dataframe, pareto_df, key)

    return pareto_df

    
def plot_all_results(dataframe, pareto_dataframe):
    """
    Level 1 : Plot per run results
    """
    # Plot rd-curves
    plot_pareto_figs_all(pareto_dataframe)


def plot_settings(dataframe, pareto_dataframe, key):
    if key in ["IT-DL-PCC", "G-PCC", "DeepPCC"]:
        return # No pareto fronts for this

    metrics = ["pcqm", "bpp", "sym_y_psnr", "sym_p2p_psnr"]
    for sequence in dataframe["sequence"].unique():
        df = dataframe[dataframe["sequence"]== sequence].sort_values(by=["q_a", "q_g"])
        pareto_df = pareto_dataframe[pareto_dataframe["sequence"]== sequence]

        x = df["q_a"].values
        y = df["q_g"].values
        if key in "YOGA":
            x = ((x - 1) / 19)
            y = ((y - 1) / 19) 
            
        X, Y = np.meshgrid(np.linspace(x.min(), x.max(), len(x)), np.linspace(y.min(), y.max(), len(y)))

        for metric in metrics:
            z = df[metric].values
            z_interp = griddata((x, y), z, (X,Y), method="linear")

            fig = plt.figure(figsize=(2.5, 2))
            ax = fig.add_subplot(111)

            ranges = {
                "bpp": [0.0, 1.8], "pcqm": [0.986, 0.998], "sym_y_psnr": [22, 40], "sym_yuv_psnr": [26, 46], "sym_p2p_psnr": [64, 80],
            }

            num_levels = {"bpp": 0.1, "pcqm": 0.002, "sym_yuv_psnr": 5, "sym_y_psnr": 2, "sym_p2p_psnr": 2}
            num_levels_bar = {"bpp": 0.2, "pcqm": 0.002, "sym_yuv_psnr": 5, "sym_y_psnr": 2, "sym_p2p_psnr": 4}
            min, max = ranges[metric]
            step = num_levels[metric]
            bar_step = num_levels_bar[metric]
            levels = np.arange(min, max+step, step)
            bar_levels = np.arange(min, max+bar_step, bar_step)

            # Pareto in countour
            cs2 = ax.contourf(X, Y, z_interp, 10, levels=levels, cmap=cm.cool, extend='min')
            """
            if key in "YOGA":
                ax.plot(pareto_df["q_a"]/20, pareto_df["q_g"]/20, color="red", marker="x", clip_on=False, label="Pareto-Front")
            else: 
                ax.plot(pareto_df["q_a"], pareto_df["q_g"], color="red", marker="x", clip_on=False, label="Pareto-Front")
            """


            # Add Settings
            settings = runs[key]["bd_points"]
            my_key = None
            for k, values in datasets.items():
                if sequence in values:
                    my_key = k
            if my_key in settings:
                settings = settings[my_key]
            else:
                continue

            q_as, q_gs = [], []
            for i, (q_a, q_g) in enumerate(settings):
                if key == "YOGA":
                    q_a = (q_a - 1)/19
                    q_g = (q_g - 1)/19
                q_as.append(q_a)
                q_gs.append(q_a)
                """
                if i == 0:
                    ax.scatter(q_a, q_g, s=40, edgecolors="#003366", marker="o", facecolors="none", linewidth=2, clip_on=False, label="Select Config.")
                else:
                    ax.scatter(q_a, q_g, s=40, edgecolors="#003366", marker="o", facecolors="none", linewidth=2, clip_on=False, )
                """
            ax.plot(q_as, q_gs, color="#003366", marker="o", clip_on=False, markersize=5, label="Selected Config.")

            ax.set_xlabel(r"$q^{(A)}$")
            ax.set_ylabel(r"$q^{(G)}$", rotation=0, ha="right", va="center")
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.xaxis.set_label_coords(0.5, -0.03)
            ax.yaxis.set_label_coords(-0.03, 0.5)
            ax.legend()

            cbar = fig.colorbar(cs2, boundaries=levels, ticks=bar_levels)
            cbar.ax.set_ylabel(metric_labels[metric])
            
            ax.tick_params(axis='both', which='major', )
            cbar.ax.tick_params(axis='both', which='major', )

            fig.tight_layout()
            path = os.path.join(plots, key, "single_{}_{}.pdf".format(metric, sequence))
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)

            # Plot pareto vs. fixed config
            fig = plt.figure(figsize=(2.5, 2))
            ax = fig.add_subplot(111)

            filtered_data = filter_config_points(df, settings)
            ax.plot(pareto_df["bpp"], pareto_df[metric], color='black', label="Pareto-Front")

            bpp = filtered_data["bpp"]
            y_data = filtered_data[metric]

            bjonte_model = Bjontegaard_Model(bpp, y_data)
            x_scat, y_scat, x_dat, y_dat = bjonte_model.get_plot_data()

            ax.plot(x_dat, y_dat, 
                    label=runs[key]["label"],
                    linestyle=runs[key]["linestyles"],
                    color=runs[key]["colors"])
            ax.scatter(x_scat, y_scat, 
                    marker=runs[key]["markers"],
                    color=runs[key]["colors"])
            ax.set_xlabel(r"bpp")
            ax.set_ylabel(metric_labels[metric])
            ax.tick_params(axis='both', which='major')
            ax.xaxis.set_label_coords(0.5, -0.12)
            ax.yaxis.set_label_coords(-0.2, 0.5)
            legend = ax.legend()

            ax.grid(visible=False)
            path = os.path.join(plots, key, "rd-pareto_vs_fixed_{}_{}.pdf".format(metric, sequence))
            fig.subplots_adjust(bottom=style.bottom, top=style.top, left=style.left, right=style.right)
            fig.savefig(path)

            plt.close(fig)



def plot_pareto_figs_single(dataframe, key):
    for sequence in dataframe["sequence"].unique():
        df = dataframe[dataframe["sequence"]== sequence]
        bpp = df["bpp"].values

        for metric in metrics:
            y = df[metric].values

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(bpp, y)
            ax.set_xlabel("bpp")
            ax.set_ylabel(metric)
            ax.grid(visible=True)
            ax.tick_params(axis='both', which='major', )

            path = os.path.join(plots, key, "rd-pareto_{}_{}.pdf".format(metric, sequence))
            #fig.tight_layout()
            fig.subplots_adjust(bottom=bottom, top=top, left=left, right=right)
            fig.savefig(path)
            #fig.savefig(path, bbox_inches="tight")
            plt.close(fig)


def plot_pareto_figs_all(pareto_dataframe):
    """
    All figures as used in the publication
    """
    for metric in metrics:
        figs = {}

        for method, df in pareto_dataframe.items():
            for sequence in df["sequence"].unique():
                # Prepare figure
                if sequence in figs.keys():
                    fig, ax = figs[sequence]
                else:
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    figs[sequence] = (fig, ax)

                data = df[df["sequence"]== sequence]

                # Filter data
                (bpp_min, bpp_max) = pareto_ranges[sequence]["bpp"]
                (y_min, y_max) = pareto_ranges[sequence][metric]
                filtered_data = data[(data['bpp'] >= bpp_min) & (data['bpp'] <= bpp_max) & (data[metric] >= y_min) & (data[metric] <= y_max)]
                bpp = filtered_data["bpp"]
                y = filtered_data[metric]

                ax.plot(bpp, y, 
                        label=labels[method],
                        linestyle=linestyles[method],
                        color=run_colors[method])
                ax.set_xlabel(r"bpp")
                ax.set_ylabel(metric_labels[metric])
                ax.tick_params(axis='both', which='major', )

        for key, items in figs.items():
            fig, ax = items
            ax.legend()
            ax.grid(visible=True)
            #fig.tight_layout()
            path = os.path.join(plots, "all", "rd-pareto_{}_{}.pdf".format(metric, key))

            fig.subplots_adjust(bottom=bottom, top=top, left=left, right=right)
            fig.savefig(path)
            #fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
        
def filter_config_points(data, config):
    tolerance = 1e-5

    mask = np.full(len(data), False)
    for q_g_test, q_a_test in config:
        # Create a mask for the current tuple
        tuple_mask = np.logical_and(
            np.isclose(data['q_a'], q_a_test, atol=tolerance),
            np.isclose(data['q_g'], q_g_test, atol=tolerance)
        )
        mask = np.logical_or(mask, tuple_mask)

    filtered_data = data[mask]
    return filtered_data

def plot_rd_figs_all(dataframes, folder):
    """
    All figures as used in the publication
    """
    for metric in metrics:
        figs = {}
        # Loop through results
        for method, df in dataframes.items():
            for sequence in df["sequence"].unique():
                # Prepare figure
                if sequence in figs.keys():
                    fig, ax = figs[sequence]
                else:
                    fig = plt.figure(figsize=(2.5, 2))
                    ax = fig.add_subplot(111)
                    figs[sequence] = (fig, ax)

                data = df[df["sequence"]== sequence]
                settings = runs[method]["bd_points"]

                my_key = None
                for key, values in datasets.items():
                    if sequence in values:
                        my_key = key
                if my_key in settings:
                    settings = settings[my_key]
                else:
                    continue

                filtered_data = filter_config_points(data, settings)

                bpp = filtered_data["bpp"]
                y = filtered_data[metric]

                bjonte_model = Bjontegaard_Model(bpp, y)
                x_scat, y_scat, x_dat, y_dat = bjonte_model.get_plot_data()

                ax.plot(x_dat, y_dat, 
                        label=runs[method]["label"][folder],
                        linestyle=runs[method]["linestyles"],
                        color=runs[method]["colors"])
                ax.scatter(x_scat, y_scat, 
                        marker=runs[method]["markers"],
                        color=runs[method]["colors"])
                ax.set_xlabel(r"bpp")
                ax.set_ylabel(metric_labels[metric])
                ax.tick_params(axis='both', which='major')
                if metric == "pcqm":
                    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.002))
                    ax.yaxis.set_label_coords(-0.2, 0.5)
                else:
                    ax.yaxis.set_label_coords(-0.12, 0.5)
                ax.xaxis.set_label_coords(0.5, -0.12)
                

        for key, items in figs.items():
            fig, ax = items
            ax.legend()
            ax.grid(visible=True)
            path = os.path.join(plots, folder, "rd-config_{}_{}.pdf".format(metric, key))
            fig.subplots_adjust(bottom=style.bottom, top=style.top, left=style.left, right=style.right)
            fig.savefig(path)

            plt.close(fig)


def compute_bd_deltas(dataframes, references, test, dir):
    results = []
    for ref in references:
        for metric in metrics:
            # Get G-PCC config for BD Points
            ref_data = dataframes[ref]
            test_data = dataframes[test]
            for sequence in ref_data["sequence"].unique():
                # Get Reference data
                ref_df = ref_data[ref_data["sequence"]== sequence]

                ref_settings = runs[ref]["bd_points"]
                key = None
                for k, values in datasets.items():
                    if sequence in values:
                        key = k
                if key in ref_settings:
                    ref_settings = ref_settings[key]
                else:
                    continue
                filtered_ref = filter_config_points(ref_df, ref_settings)

                # Get test data
                test_df = test_data[test_data["sequence"]== sequence]
                test_settings = runs[test]["bd_points"]
                key = None
                for k, values in datasets.items():
                    if sequence in values:
                        key = k
                if key in test_settings:
                    test_settings = test_settings[key]
                else:
                    continue
                filtered_test = filter_config_points(test_df, test_settings)


                bpp = filtered_ref["bpp"]
                y = filtered_ref[metric]
                ref_model = Bjontegaard_Model(bpp, y)

                bpp = filtered_test["bpp"]
                y = filtered_test[metric]
                test_model = Bjontegaard_Model(bpp, y)

                delta = Bjontegaard_Delta()
                psnr_delta = delta.compute_BD_PSNR(ref_model, test_model)
                rate_delta = delta.compute_BD_Rate(ref_model, test_model)

                results.append({
                    "reference": ref,
                    "test": test,
                    "sequence": sequence,
                    "metric": metric,
                    "psnr_delta": psnr_delta,
                    "rate_delta": rate_delta
                })

    results_df = pd.DataFrame(results)
    output_path = os.path.join(plots, dir, f"bd_deltas_{test}.csv")
    results_df.to_csv(output_path, index=False)


def get_pareto_df(dataframe):
    pareto_dataframe = []
    for sequence in dataframe["sequence"].unique():
        df = dataframe[dataframe["sequence"]== sequence].sort_values(by=["bpp"])

        pareto_front = []
        max_pcqm = 0.0

        # Iterate through the sorted DataFrame
        for index, row in df.iterrows():
            pcqm = row['pcqm']
    
            if pcqm >= max_pcqm:
                max_pcqm = pcqm
                pareto_front.append(index)

        # Create a new DataFrame for the Pareto front
        pareto_dataframe.append(df.loc[pareto_front])
    pareto_dataframe = pd.concat(pareto_dataframe, ignore_index=True)
    return pareto_dataframe

def load_csvs(keys):
    data = {}
    for key in keys:
        data_path = os.path.join(path, key, "test.csv")
        data[key] = pd.read_csv(data_path)

        # Preprocessing
        data[key]["pcqm"] = 1 - data[key]["pcqm"]

        # Average per data set
        averaged_rows = []
        for testset, sequences in datasets.items():
            # Filter the DataFrame to only include rows with these sequences
            filtered_df = data[key][data[key]['sequence'].isin(sequences)]

            # Group by unique `q_a` and `q_g` configurations and compute the mean for each group
            grouped = filtered_df.groupby(['q_a', 'q_g']).mean(numeric_only=True).reset_index()
            
            # Assign the sequence name to each group average
            grouped['sequence'] = testset
            
            # Append the grouped result to the list of averaged rows
            averaged_rows.append(grouped)

        # Convert the list of averaged rows to a DataFrame
        averaged_df = pd.concat(averaged_rows, ignore_index=True)
        data[key] = pd.concat([data[key], averaged_df], ignore_index=True)
    return data


def compute_times(data):
    # Computes the times
    summary_data = []
    for key, results in data.items():
        if key == "YOGA":
            continue

        for sequence in results["sequence"].unique():
            test_sequences = ["loot", "longdress", "soldier", "redandblack"]
            if sequence not in test_sequences:
                continue
            

            if key == "G-PCC":
                #process per rate
                rates = [0.125, 0.25, 0.5, 0.75]
                for rate in rates:
                    t_compress = results[(results["sequence"] == sequence) & (results["q_g"] == rate)]["t_compress"]
                    t_decompress = results[(results["sequence"] == sequence) & (results["q_g"] == rate)]["t_decompress"]

                    conf_compress = st.t.interval(0.95, len(t_compress-1), loc=np.mean(t_compress), scale=st.sem(t_compress))
                    conf_decompress = st.t.interval(0.95, len(t_decompress-1), loc=np.mean(t_decompress), scale=st.sem(t_decompress))

                    summary_data.append([key, sequence, rate, np.mean(t_compress), np.mean(t_compress) - conf_compress[0], np.mean(t_decompress), np.mean(t_decompress) - conf_decompress[0]])
            else:
                # process all
                t_compress = results[results["sequence"] == sequence]["t_compress"]
                t_decompress = results[results["sequence"] == sequence]["t_decompress"]
                conf_compress = st.t.interval(0.95, len(t_compress-1), loc=np.mean(t_compress), scale=st.sem(t_compress))
                conf_decompress = st.t.interval(0.95, len(t_decompress-1), loc=np.mean(t_decompress), scale=st.sem(t_decompress))

                summary_data.append([key, sequence, None, np.mean(t_compress), np.mean(t_compress) - conf_compress[0], np.mean(t_decompress), np.mean(t_decompress) - conf_decompress[0]])


        # Calculate per sequence mean per key and rate
        results = results[results["sequence"].isin(test_sequences)]
        if key == "G-PCC":
            rates = [0.125, 0.25, 0.5, 0.75]
            for rate in rates:
                    t_compress_seq_rate = results[results["q_g"] == rate]["t_compress"]
                    t_decompress_seq_rate = results[results["q_g"] == rate]["t_decompress"]

                    mean_t_compress_seq_rate = np.mean(t_compress_seq_rate)
                    mean_t_decompress_seq_rate = np.mean(t_decompress_seq_rate)

                    summary_data.append([key, "combined", rate, mean_t_compress_seq_rate, np.nan, mean_t_decompress_seq_rate, np.nan])

        t_compress_seq = results["t_compress"]
        t_decompress_seq = results["t_decompress"]

        mean_t_compress_seq = np.mean(t_compress_seq)
        mean_t_decompress_seq = np.mean(t_decompress_seq)

        summary_data.append([key, "combined", None, mean_t_compress_seq, np.nan , mean_t_decompress_seq, np.nan])

    summary_df = pd.DataFrame(summary_data, columns=["key", "sequence", "rate", "mean_t_compress", "conf_compress", "mean_t_decompress", "conf_decompress"])

    print(summary_df)
if __name__ == "__main__":
    plot_experiments()

