import matplotlib
import matplotlib.pyplot as plt

# Settings 
matplotlib.use("template")
# General settings
plt.rcParams.update({
    'font.size': 8,                    # General font size for labels and titles
    'font.family': 'serif',            # Use a serif font
    'axes.titlesize': 8,               # Font size for axes titles
    'axes.labelsize': 8,               # Font size for axis labels
    'xtick.labelsize': 7,              # Font size for x-tick labels
    'ytick.labelsize': 7,              # Font size for y-tick labels
    'lines.linewidth': 1.5,              # Default line width
    'lines.markersize': 3,             # Default marker size
    'legend.fontsize': 8,              # Font size for legend
    'legend.frameon': True,            # Legend frame on
    'legend.framealpha': 1.0,          # Legend frame opacity
    'axes.axisbelow': True,            # Place grid and ticks below plot elements
    'savefig.dpi': 300,                # Resolution for saved figures
    'savefig.format': 'pdf',           # File format for saving figures
    'pdf.fonttype': 42,                # Use Type 42 (TrueType) fonts in PDF
    'ps.fonttype': 42                  # Use Type 42 (TrueType) fonts in PS
})


matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['pdf.fonttype'] = 42

linestyles = ["solid", "dashdot", "dashed", (0, (3, 1, 1, 1)), "dotted", ]

colors = [ "#003366", "#e31b23", "#787878", "#1a9e00", "#03b5fc", "green"] #RPTH Palette

markers = [ "o", "v", "s", "P", "X"] 
markers = [ "x", "x", "x", "x", "x"] 




top = .97
bottom = .16
left = .22
right = .97
runs = {
    "G-PCC": {
        "label":
            {"sota_comparison": "G-PCC"},
        "bd_points": {"8iVFBv2" : [(0.5, 40), (0.75, 34), (0.875, 28), (0.9375, 22)], 
                      "Owlii": [(0.25, 40), (0.5, 34), (0.75, 28), (0.875, 22)]},
        "pareto_ranges": {"bpp": [0.0, 2.0], "pcqm": [0.98, 0.9975], "sym_y_psnr": [22, 30], "sym_yuv_psnr": [22, 1.00], "sym_p2p_psnr": [60, 70]},
        "colors": colors[2],
        "linestyles": linestyles[2],
        "markers": markers[2],
    },
    "DeepPCC": {
        "label": 
            {"sota_comparison": "DeepPCC",
             "ablation_fixed": "DeepPCC"},
        "bd_points": {"8iVFBv2": [(1, 1), (2, 2), (3, 3), (4, 4)], },
        "pareto_ranges": {"bpp": [0.0, 2.0], "pcqm": [0.98, 0.9975], "sym_y_psnr": [22, 30], "sym_yuv_psnr": [22, 1.00], "sym_p2p_psnr": [60, 70]},
        "colors": colors[1],
        "linestyles": linestyles[1],
        "markers": markers[1],
    },
    "IT-DL-PCC": {
        "label": 
            {"sota_comparison": "IT-DL-PCC"},
        "bd_points": {"8iVFBv2" :[(0.001, 0.0), (0.002, 0.0), (0.004, 0.0), (0.0005, 0.0)],
                      "Owlii" :[(0.001, 0.0), (0.002, 0.0), (0.004, 0.0), (0.0005, 0.0)]},
        "pareto_ranges": {"bpp": [0.0, 2.0], "pcqm": [0.98, 0.9975], "sym_y_psnr": [22, 30], "sym_yuv_psnr": [22, 1.00], "sym_p2p_psnr": [60, 70]},
        "colors": colors[3],
        "linestyles": linestyles[3],
        "markers": markers[3],
    },
    "CVPR_inverse_scaling": {
        "label": 
            {"sota_comparison": "Ours",
             "ablation_scaling": "symmetric gain",
             "ablation_loss": "MSE",
             "ablation_fixed": "Ours (1 Model)"},
        "bd_points": {
            "8iVFBv2": [(0.1, 0.1), (0.2, 0.2), (0.4, 0.4),  (1.0, 1.0)],
            "Owlii": [(0.1, 0.1), (0.2, 0.2), (0.4, 0.4),  (1.0, 1.0)]},
        "pareto_ranges": {"bpp": [0.0, 2.0], "pcqm": [0.98, 0.9975], "sym_y_psnr": [22, 30], "sym_yuv_psnr": [22, 1.00], "sym_p2p_psnr": [60, 70]},
        "colors": colors[0],
        "linestyles": linestyles[0],
        "markers": markers[0],
    },
    "Final_L2_GDN_scale_rescale_ste_offsets_inverse_nn_vbr_btlnk" : {
        "label": 
            {"sota_comparison": "Ours",
             "ablation_scaling": "asymmetric gain",
             "ablation_loss": "MSE"},
        "bd_points": {
            "8iVFBv2": [(0.1, 0.1), (0.2, 0.2), (0.4, 0.4),  (1.0, 1.0)],
            "Owlii": [(0.1, 0.1), (0.2, 0.2), (0.4, 0.4),  (1.0, 1.0)]},
        "pareto_ranges": {"bpp": [0.0, 2.0], "pcqm": [0.98, 0.9975], "sym_y_psnr": [22, 30], "sym_yuv_psnr": [22, 1.00], "sym_p2p_psnr": [60, 70]},
        "colors": colors[1],
        "linestyles": linestyles[1],
        "markers": markers[1],
    },
    "YOGA" : {
        "label": 
            {"sota_comparison": "YOGA"},
        "bd_points": {
            "8iVFBv2" : [ (8, 3), (12, 6), (18, 10),  (20, 20)],
        },
        "pareto_ranges": {"bpp": [0.0, 2.0], "pcqm": [0.98, 0.9975], "sym_y_psnr": [22, 30], "sym_yuv_psnr": [22, 1.00], "sym_p2p_psnr": [60, 70]},
        "colors": colors[4],
        "linestyles": linestyles[4],
        "markers": markers[4],
    },
    "Final_L2_GDN_scale_rescale_ste_offsets_shepard_2" : {
        "label": 
            {"sota_comparison": "Ours",
             "ablation_scaling": "asymmetric gain",
             "ablation_loss": "MSE+IWD"},
        "bd_points": {
            "8iVFBv2": [(0.1, 0.1), (0.2, 0.2), (0.4, 0.4),  (1.0, 1.0)],
            "Owlii": [(0.1, 0.1), (0.2, 0.2), (0.4, 0.4),  (1.0, 1.0)]},
        "pareto_ranges": {"bpp": [0.0, 2.0], "pcqm": [0.98, 0.9975], "sym_y_psnr": [22, 30], "sym_yuv_psnr": [22, 1.00], "sym_p2p_psnr": [60, 70]},
        "colors": colors[0],
        "linestyles": linestyles[0],
        "markers": markers[0],
    },
    "CVPR_inverse_scaling_shepard" : {
        "label": 
            {"sota_comparison": "Ours",
             "ablation_scaling": "symmetric gain",
             "ablation_loss": "MSE+IWD"},
        "bd_points": {
            "8iVFBv2": [(0.1, 0.1), (0.2, 0.2), (0.4, 0.4),  (1.0, 1.0)],
            "Owlii": [(0.1, 0.1), (0.2, 0.2), (0.4, 0.4),  (1.0, 1.0)]},
        "pareto_ranges": {"bpp": [0.0, 2.0], "pcqm": [0.98, 0.9975], "sym_y_psnr": [22, 30], "sym_yuv_psnr": [22, 1.00], "sym_p2p_psnr": [60, 70]},
        "colors": colors[1],
        "linestyles": linestyles[1],
        "markers": markers[1],
    },
    "CVPR_inverse_scaling_fixed" : {
        "label": 
            { "ablation_fixed": "Fixed (4 Models)"},
        "bd_points": {
            "8iVFBv2": [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0),  (4.0, 4.0)],
            "Owlii": [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0),  (4.0, 4.0)]},
        "pareto_ranges": {"bpp": [0.0, 2.0], "pcqm": [0.98, 0.9975], "sym_y_psnr": [22, 30], "sym_yuv_psnr": [22, 1.00], "sym_p2p_psnr": [60, 70]},
        "colors": colors[1],
        "linestyles": linestyles[1],
        "markers": markers[1],
    },
}


metric_labels = {
    "bpp" : r"bpp",
    "pcqm" : r"$1 -$ PCQM",
    "sym_y_psnr" : r"Y-PSNR [dB]",
    "sym_yuv_psnr" : r"YUV-PSNR [dB]",
    "sym_p2p_psnr" : r"D1-PSNR [dB]",
    "sym_d2_psnr" : r"D2-PSNR [dB]",
}
