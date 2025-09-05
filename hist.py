import sentiment_from_csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable


def histo():
    df = sentiment_from_csv.run_sentiment_analysis(source_df, text_column="Text")
    # Extract net sentiment for the histogram
    hist = df['net_sentiment']

    # Create a custom colormap: red for [-1, 0], green for [0, 1]
    colors = [(1, 0, 0), (1, 1, 1), (0, .5, 0)]  # Red to white to green
    nodes = [0, 0.5, 1]  # Normalize -1 to 0, 0 to 0.5, and 1 to 1
    custom_cmap = LinearSegmentedColormap.from_list("red_to_green", list(zip(nodes, colors)))

    # Explicitly create a figure and axes
    fig, ax = plt.subplots()

    # Create the histogram data on the ax
    counts, bins, patches = ax.hist(hist, bins=20, edgecolor='black', orientation='vertical')

    # Normalize the bin centers to the range [-1, 1]
    bin_centers = 0.5 * (bins[:-1] + bins[1:])  # Calculate the center of each bin
    norm = Normalize(vmin=-1, vmax=1)

    # Apply the gradient colors to the patches
    for center, patch in zip(bin_centers, patches):
        color = custom_cmap(norm(center))  # Map bin center to the colormap
        patch.set_facecolor(color)

    # Add a colorbar
    sm = ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])  # Empty array for colorbar mapping
    cbar = fig.colorbar(sm, ax=ax)  # Explicitly link the colorbar to the ax
    cbar.set_label("Net Sentiment (-1 to 1)")

    # Add titles and labels
    # ax.set_title("Net Sentiment Distribution")
    # ax.set_ylabel("Frequency")
    # ax.set_xlabel("Net Sentiment")

    # Show the plot
    plt.show()