"""
Simple styling used for matplotlib figures
"""

from matplotlib import pyplot as plt

# Configuration settings to help visibility on small screen / prints
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['figure.titlesize'] = 15
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['figure.figsize'] = [.8 * 6.4, .8 * 4.8]
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.columnspacing'] = 1.8
plt.rcParams['legend.handlelength'] = 1.5
plt.rcParams['legend.handletextpad'] = 0.5

# Utility functions
def light_axis():
    "Hide the top and right spines"
    ax = plt.gca()
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    plt.xticks(())
    plt.yticks(())
    plt.subplots_adjust(left=.01, bottom=.01, top=.99, right=.99)

def no_axis():
    plt.axis('off')
    plt.subplots_adjust(left=.0, bottom=.0, top=1, right=1)
