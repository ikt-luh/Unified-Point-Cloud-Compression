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
    'lines.linewidth': 2,              # Default line width
    'lines.markersize': 1,             # Default marker size
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

linestyles = ["solid", "dashdot", "dashed", "dotted", (0, (3, 1, 1, 1))]

colors = [ "#003366", "#e31b23", "#787878", "#1a9e00", "#03b5fc", "green"] #RPTH Palette

markers = [ "o", "v", "s", "P", "X"] 