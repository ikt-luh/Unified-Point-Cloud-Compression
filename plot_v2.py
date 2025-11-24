import os
import yaml
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

# Setup
path = "./results"
plots = "./plot/figures"
experiment_config = "./plot/configs/rd_evals.yaml"
metrics = ["pcqm", "sym_y_psnr", "sym_p2p_psnr", "sym_d2_psnr", "sym_yuv_psnr"]
plot_areas = ["Main", "Ablation_Inverse_nn"]


def load_yaml(filepath):
    with open(filepath, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_dirs():
    pass

def load_dataframe(method):
    path = os.path.join("results", method, "test.csv")
    df = pd.read_csv(path)
    return df

def filter_df_by_config(df, rate_config, pointcloud):
    rate_config_df = pd.DataFrame(rate_config)

    filtered_df = df[df["sequence"] == pointcloud]

    # Round float columns in both dataframes for easier search (only key columns)
    float_precision = 4
    float_cols = [col for col in rate_config.keys() if df[col].dtype.kind == 'f']
    filtered_df.loc[:, float_cols] = filtered_df[float_cols].round(float_precision)
    rate_config_df.loc[:, float_cols] = rate_config_df[float_cols].round(float_precision)

    result = filtered_df.merge(rate_config_df, how="inner", on=list(rate_config.keys()))
    return result

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


def plot_experiments():
    """
    Level 0 : Plot all results
    """
    # Load the plot config
    plot_config = load_yaml(experiment_config)

    # Load all dataframes
    methods = plot_config["settings"]["methods"]
    dataframes = {}
    rate_configs = {}
    for method in methods:
        df = load_dataframe(method)
        dataframes[method] = df

        rate_config_path = os.path.join("results", method, "plot_config.yaml")
        rate_config = load_yaml(rate_config_path)
        rate_configs[method] = rate_config 

    # Plot each study
    plot_configs = plot_config["plots"]
    for key in plot_configs.keys():
        bd_models = plot_rd_figs(key, plot_configs, rate_configs, dataframes)
        compute_bd_deltas(bd_models, reference="Main", file_key=key)

        bd_models_avg = plot_rd_figs_avg(key, plot_configs, rate_configs, dataframes)
        compute_bd_deltas(bd_models_avg, reference="Main", file_key="{}_{}".format(key, "avg"))


    for key in plot_areas:
        plot_area_figs(dataframes[key], key, "longdress")

    # Plot legend (hacky)
    plot_wide_legend(
        method_configs=rate_configs,                
        plot_configs=plot_configs
    )


    
def plot_rd_figs(key, plot_configs, method_configs, dataframes):
    """
    All figures as used in the publication
    """
    plot_config = plot_configs[key]
    groups = plot_config["groups"]
    methods = plot_config["methods"]

    bd_models = {}
    times = []
    for metric in metrics:
        bd_models[metric] = {}
        for group in groups:
            pointclouds = groups[group]

            for pointcloud in pointclouds:
                bd_models[metric][pointcloud] = {}
                # Prepare figure
                fig = plt.figure(figsize=(2.5, 2))
                ax = fig.add_subplot(111)

                # Plot all methods:
                for method in methods:
                    # Load the rate config for this eval
                    method_config = method_configs[method][key]
                    if pointcloud not in method_config["rate-points"].keys():
                        continue
                    rate_config = method_config["rate-points"][pointcloud]
                    plot_style = method_config["info"]

                    # Filter the dataframe by the specified rate points
                    filtered_df = filter_df_by_config(dataframes[method], rate_config, pointcloud)

                    # Get values
                    bpp = filtered_df["bpp"]
                    y = filtered_df[metric]
                    if metric == "pcqm":
                        y = 1 - y # PCQM is reported in 1-PCQM
                        
                        # Average timings 
                        if "t_compress" in filtered_df.columns:
                            t_comp = np.mean(filtered_df["t_compress"])
                            t_decomp = np.mean(filtered_df["t_decompress"])
                            times.append({"method": method,
                                        "pointcloud": pointcloud,
                                        "t_comp": t_comp,
                                        "t_decomp": t_decomp})
                    
                    print("{} {} {}".format(method, pointcloud, len(bpp)))

                    # Fit Bjontegaard model and get plot data
                    bjonte_model = Bjontegaard_Model(bpp, y)
                    bd_models[metric][pointcloud][method] = bjonte_model
                    x_scat, y_scat, x_dat, y_dat = bjonte_model.get_plot_data()

                    # Plot and scatter curves
                    ax.plot(x_dat, y_dat, 
                            label=plot_style["label"],
                            linestyle=plot_style["linestyle"],
                            linewidth=0.8,
                            alpha=0.8,
                            color=plot_style["color"])
                    ax.scatter(x_scat, y_scat, 
                            marker=plot_style["marker"],
                            s=5,
                            color=plot_style["color"])
                    
                # Plot labeling
                ax.set_ylabel(metric_labels[metric])

                ax.set_xlabel(r"bpp")
                ax.tick_params(axis='both', which='major')
                if metric == "pcqm":
                    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.005))
                    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.001))
                else:
                    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
                    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
                    
                # Coords
                ax.yaxis.set_label_coords(-0.12, 0.5)
                ax.xaxis.set_label_coords(0.5, -0.12)
                ax.set_xlim(left=0)
                    
                # Finish Plot
                ax.legend(loc=4, labelspacing=0.3)
                ax.grid(visible=True)

                # Save plot
                base_path = os.path.join(plots, key)
                if not os.path.exists(base_path):
                    os.makedirs(base_path)

                path = os.path.join(base_path, "rd-config_{}_{}.pdf".format(metric, pointcloud))

                fig.subplots_adjust(bottom=style.bottom, top=style.top, left=style.left, right=style.right)
                fig.savefig(path)

                # Cleanup
                plt.close(fig)

    times_df = pd.DataFrame(times)
    output_path = os.path.join(plots, "times", f"timings_{key}.csv")
    times_df.to_csv(output_path, index=False)
    return bd_models

def plot_wide_legend(method_configs, plot_configs, keys=("main_8i","main_jpeg"),
                     filename="legend.pdf", dpi=300):
    # preserve method order as it appears in the plot configs
    ordered_methods = []
    seen = set()
    for k in keys:
        if k not in plot_configs:
            continue
        for m in plot_configs[k].get("methods", []):
            if m not in seen:
                ordered_methods.append(m)
                seen.add(m)

    fig, ax = plt.subplots(figsize=(8, 0.6))  # wide, short strip
    handles, labels = [], []

    for m in ordered_methods:
        # pick the first key where this method has style info
        style = None
        for k in keys:
            if m in method_configs and k in method_configs[m]:
                style = method_configs[m][k]["info"]
                break
        if style is None:
            continue  # method missing style; skip

        line, = ax.plot([], [],
                        label=style["label"],
                        linestyle=style["linestyle"],
                        linewidth=1,
                        color=style["color"],
                        marker=style["marker"],
                        markersize=5)
        handles.append(line)
        labels.append(style["label"])

    ax.axis("off")

    legend = ax.legend(handles=handles, labels=labels,
                       loc="center", frameon=False,
                       ncol=len(handles),            # single row across the page
                       handletextpad=0.5, columnspacing=1.0)

    fig.canvas.draw()
    path = os.path.join(plots, filename)
    fig.savefig(path)
    plt.close(fig)



def plot_rd_figs_avg(key, plot_configs, method_configs, dataframes):
    """
    All figures as used in the publication
    """
    plot_config = plot_configs[key]
    groups = plot_config["groups"]
    methods = plot_config["methods"]

    bd_models = {}
    for metric in metrics:
        bd_models[metric] = {}
        for group in groups:
            bd_models[metric][group] = {}
            pointclouds = groups[group]

            # Prepare figure
            fig = plt.figure(figsize=(3, 2))
            ax = fig.add_subplot(111)

            # Plot all methods:
            for method in methods:
                method_bpp, method_metric = [], []
                for pointcloud in pointclouds:
                    # Load the rate config for this eval
                    method_config = method_configs[method][key]
                    if pointcloud not in method_config["rate-points"].keys():
                        continue
                    rate_config = method_config["rate-points"][pointcloud]
                    plot_config = method_config["info"]

                    # Filter the dataframe by the specified rate points
                    filtered_df = filter_df_by_config(dataframes[method], rate_config, pointcloud)
                    filtered_df = filtered_df.sort_values("bpp")

                    # Get values
                    bpp = filtered_df["bpp"]
                    y = filtered_df[metric]
                    if metric == "pcqm":
                        y = 1 - y # PCQM is reported in 1-PCQM
                    method_bpp.append(bpp)
                    method_metric.append(y)
                    
                method_bpp = np.stack(method_bpp)
                method_metric = np.stack(method_metric)
                avg_bpp = np.mean(method_bpp, axis=0)
                avg_metric = np.mean(method_metric, axis=0)

                # Fit Bjontegaard model and get plot data
                bjonte_model = Bjontegaard_Model(avg_bpp, avg_metric)
                x_scat, y_scat, x_dat, y_dat = bjonte_model.get_plot_data()
                bd_models[metric][group][method] = bjonte_model

                # Plot and scatter curves
                ax.plot(x_dat, y_dat, 
                        label=plot_config["label"],
                        linestyle=plot_config["linestyle"],
                        linewidth=1,
                        alpha=0.8, #0.8
                        color=plot_config["color"])
                ax.scatter(x_scat, y_scat, 
                        marker=plot_config["marker"],
                        s=15,
                        color=plot_config["color"])
                    
            # Plot labeling
            ax.set_xlabel(r"bpp")
            ax.set_ylabel(metric_labels[metric])
            ax.tick_params(axis='both', which='major')
            if metric == "pcqm":
                # Set y lims to upper 1
                ax.set_ylim(ax.get_ylim()[0], 1.0)

                ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.001))
                ax.yaxis.set_label_coords(-0.18, 0.5)
            else:
                ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
                ax.yaxis.set_label_coords(-0.15, 0.5)

            if group in ["sparse", "dense"]:
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
            else:
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

            ax.xaxis.set_label_coords(0.5, -0.14)
            ax.set_xlim(left=0)

                    
            # Finish Plot
            if key not in ["main_8i", "main_jpeg"]:
                ax.legend(loc=4, labelspacing=0.3, bbox_to_anchor=(1,0)) # Was no font size

            ax.grid(visible=True)

            # Save plot
            base_path = os.path.join(plots, key)
            if not os.path.exists(base_path):
                os.makedirs(base_path)

            path = os.path.join(base_path, "rd-config_avg_{}_{}.pdf".format(metric, group))

            fig.subplots_adjust(bottom=style.bottom, top=style.top, left=style.left, right=style.right)
            fig.savefig(path)

            # Cleanup
            plt.close(fig)


    
    return bd_models
    

def plot_area_figs(dataframe, method, pointcloud):
    df = dataframe[dataframe["sequence"]==pointcloud].sort_values(by=["q_a", "q_g"])
    df = df[df["scale_factor"]== 1].sort_values(by=["q_a", "q_g"])

    x = df["q_a"].values
    y = df["q_g"].values
            
    X, Y = np.meshgrid(np.linspace(x.min(), x.max(), len(x)), np.linspace(y.min(), y.max(), len(y)))

    area_metrics = ["bpp", "pcqm", "sym_yuv_psnr", "sym_y_psnr", "sym_p2p_psnr", "sym_d2_psnr"]
    for metric in area_metrics:
        z = df[metric].values
        if metric == "pcqm":
            z = 1 - z

        z_interp = griddata((x, y), z, (X,Y), method="linear")

        fig = plt.figure(figsize=(2.5, 2))
        ax = fig.add_subplot(111)

        ranges = {
            "bpp": [0.0, 2.4], "pcqm": [0.986, 0.998], "sym_y_psnr": [22, 40], "sym_yuv_psnr": [26, 46], "sym_p2p_psnr": [64, 80], "sym_d2_psnr": [64, 84],
        }

        num_levels = {"bpp": 0.2, "pcqm": 0.002, "sym_yuv_psnr": 5, "sym_y_psnr": 1, "sym_p2p_psnr": 1, "sym_d2_psnr": 1}
        num_levels_bar = {"bpp": 0.2, "pcqm": 0.002, "sym_yuv_psnr": 5, "sym_y_psnr": 2, "sym_p2p_psnr": 4, "sym_d2_psnr": 4}
        min, max = ranges[metric]
        step = num_levels[metric]
        bar_step = num_levels_bar[metric]
        levels = np.arange(min, max+step, step)
        bar_levels = np.arange(min, max+bar_step, bar_step)

        # Pareto in countour
        cs2 = ax.contourf(X, Y, z_interp, 10, levels=levels, cmap=cm.cool, extend='min')

        ax.contour(cs2, colors="k", levels=levels, linewidths=0.2)
        ax.grid(c='k', ls='-', alpha=0.3)

        """
        q_as, q_gs = [], []
        for i, (q_g, q_a) in enumerate(settings):
            q_as.append(q_a)
            q_gs.append(q_g)

        ax.plot(q_as, q_gs, color="#003366", marker="o", clip_on=False, markersize=5, label="Selected Config.")
        """

        ax.set_xlabel(r"$q_a$")
        ax.set_ylabel(r"$q_g$", rotation=0, ha="right", va="center")
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.xaxis.set_label_coords(0.5, -0.03)
        ax.yaxis.set_label_coords(-0.03, 0.5)
        #ax.legend()

        cbar = fig.colorbar(cs2, boundaries=levels, ticks=bar_levels)
        cbar.ax.set_ylabel(metric_labels[metric])
            
        ax.tick_params(axis='both', which='major', )
        cbar.ax.tick_params(axis='both', which='major', )

        fig.tight_layout()
        path = os.path.join(plots, "areas", "{}_{}_{}.pdf".format(method, metric, pointcloud))
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)







def compute_bd_deltas(bd_models, reference, file_key):
    results = []
    for metric, sub_dict in bd_models.items():
        for pointcloud, sub_sub_dict in sub_dict.items():
            test_model = sub_sub_dict.pop(reference)

            for method, ref_model in sub_sub_dict.items():
                print(method)

                delta = Bjontegaard_Delta()
                psnr_delta = delta.compute_BD_PSNR(ref_model, test_model)
                rate_delta = delta.compute_BD_Rate(ref_model, test_model)
                
                print(psnr_delta, rate_delta)
                results.append({
                    "metric": metric,
                    "pointCloud": pointcloud,
                    "method": method,
                    "BD-PSNR": psnr_delta,
                    "BD-Rate": rate_delta * 100 # In Percent
                })

    results_df = pd.DataFrame(results)
    output_path = os.path.join(plots, "bd_results", f"bd_deltas_{file_key}.csv")
    results_df.to_csv(output_path, index=False)

    return



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
