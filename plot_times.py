import plot.style  # optional
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

plt.rcParams.update({
    "font.size": 8,             # base font size
    "axes.labelsize": 9,        # y/x label
    "xtick.labelsize": 8,       # tick labels
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

def plot_encoding_times(csv_path):
    df = pd.read_csv(csv_path)

    # === USER SETTINGS ===
    experiment_labels = {
        "Main": "Proposed",
        "G-PCC": "G-PCC",
        "V-PCC": "V-PCC",
        "IT-DL-PCC": "IT-DL-PCC",
        "DeepPCC": "DeepPCC",
        "JPEG Pleno PCC": "JPEG Pleno PCC",
    }
    colors = {
        "Main": "#1B3A6F",
        "G-PCC": "#787878",
        "V-PCC": "#4D4D4D",
        "IT-DL-PCC": "#228B22",
        "DeepPCC": "#FF7F00",
        "JPEG Pleno PCC": "#B8860B",
    }

    bar_labels = {
        ("Main", 0): "scale=1",
        ("Main", 1): "scale=2",
        ("Main", 2): "scale=4",
        ("G-PCC", 4): "pQs=0.9375",
        ("G-PCC", 3): "pQs=0.75",
        ("G-PCC", 2): "pQs=0.5",
        ("G-PCC", 1): "pQs=0.25",
        ("V-PCC", 4): "R4",
        ("V-PCC", 3): "R3",
        ("V-PCC", 2): "R2",
        ("V-PCC", 1): "R1",
        ("IT-DL-PCC", 0): "scale=1, SR=0",
        ("IT-DL-PCC", 1): "scale=2, SR=0",
        ("IT-DL-PCC", 2): "scale=2, SR=1",
        ("IT-DL-PCC", 3): "scale=4, SR=0",
        ("IT-DL-PCC", 4): "scale=4, SR=1",
        ("DeepPCC", 0): "",
        ("JPEG Pleno PCC", 0): "",
    }


    def plot_metric(metric, label_threshold, bottom):
        fig, ax = plt.subplots(figsize=(5, 2))

        experiments = list(experiment_labels.keys())
        x = np.arange(len(experiments))  # One x-tick per method
        bar_width = 0.2

        # Organize bar entries by experiment
        exp_to_entries = defaultdict(list)
        for (exp, idx) in bar_labels.keys():
            exp_to_entries[exp].append(idx)

        for i, exp in enumerate(experiments):
            idxs = exp_to_entries.get(exp, [])
            n = len(idxs)
            offsets = np.linspace(-bar_width * (n - 1) / 2, bar_width * (n - 1) / 2, n)

            for j, idx in enumerate(idxs):
                subset = df[(df['experiment'] == exp) & (df['IDX'] == idx)][metric]
                if subset.empty:
                    continue
                mean = subset.mean()
                std = subset.std()
                n = len(subset)
                ci = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
                print(ci / mean)


                if n > 1:
                    ax.bar(
                        x[i] + offsets[j],
                        mean,
                        alpha=0.8,
                        yerr=ci,
                        capsize=bar_width*25,
                        error_kw=dict(capthick=0.01, lw=0.01) ,
                        width=bar_width,
                        color=colors[exp]
                    )
                else:
                    ax.bar(
                        x[i] + offsets[j],
                        mean,
                        alpha=0.8,
                        width=bar_width,
                        color=colors[exp]
                    )

                label = bar_labels.get((exp, idx), "")
                if label:
                    c = "black"
                    if mean > label_threshold:
                        c = "white" if exp == "V-PCC" else c
                        ax.text(
                            x[i] + offsets[j],
                            bottom,
                            label,
                            ha='left',
                            va='center',
                            rotation=90,
                            rotation_mode='anchor',
                            color=c,
                            fontsize=7
                        )
                    else:
                        ax.text(
                            x[i] + offsets[j],
                            mean * 1.1,
                            label,
                            ha='left',
                            va='center',
                            rotation=90,
                            rotation_mode='anchor',
                            color=c,
                            fontsize=7
                        )

        ax.set_xticks(x)
        ax.set_xticklabels([experiment_labels[exp] for exp in experiments], rotation=0, ha='center')
        #ax.set_xlabel("Method")
        ax.set_ylabel("Time [s]")
        #ax.set_title(f"{metric} grouped by method")
        ax.set_yscale('log')

        #ax.legend(
        #    handles=[plt.Line2D([0], [0], color=c, marker='s', linestyle='') for exp, c in exp_to_color.items()],
        #    labels=[experiment_labels[exp] for exp in experiments],
        #    title="Method"
        #)

        ax.grid(True, which='both',)
        fig.tight_layout()
        os.makedirs("plot/figures/times", exist_ok=True)
        path = os.path.join("plot/figures", "times", f"times_{metric}_by_method.pdf")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        plt.show()

    plot_metric('t_compress', 23, bottom=1)
    plot_metric('t_decompress', 5, bottom=0.125)

if __name__ == "__main__":
    plot_encoding_times("time_measurements_soldier.csv")
