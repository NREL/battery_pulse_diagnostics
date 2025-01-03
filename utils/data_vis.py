import pandas
import matplotlib.pyplot as plt
import numpy as np


def plot_segment_parts(
    ax, df, x="Segment Time, S", y="Voltage, V", line_style="-", **kwargs
):
    # reset df index
    df = df.reset_index(drop=True)
    # Search for any discontinuities in the segment times
    segment_starts = np.argwhere(
        np.diff(df["Segment Time, S"], prepend=0) < 0
    ).squeeze()
    segment_starts = np.append([0], segment_starts)
    for i, idx in enumerate(segment_starts):
        # Extract this segment part
        if i == segment_starts.size - 1:
            df_seg_part = df.loc[idx:].copy().reset_index(drop=True)
        else:
            df_seg_part = (
                df.loc[idx : segment_starts[i + 1] - 1].copy().reset_index(drop=True)
            )
        # Plot
        ax.plot(
            df_seg_part[x],
            df_seg_part[y],
            line_style,
            label=df_seg_part.loc[0, "Segment Description"],
            **kwargs
        )
    # Decorations
    ax.set_xlabel(x)
    ax.set_ylabel(y)


def plotyy(y, y_pred, color=None, title=None):
    # Customize colors to match with capacity vs cycle plot earlier
    if color is not None:
        n_series = np.unique(color).size
    else:
        n_series = 1
    cmap = plt.get_cmap("tab10", n_series)
    # Get axes and plot
    fig, ax = plt.subplots(1, 1)
    sc = ax.scatter(y.values, y_pred, c=color, cmap=cmap)
    ax.set_aspect("equal")
    plt.xlabel("Actual discharge capacity (mAh)")
    plt.ylabel("Predicted discharge capacity (mAh)")
    plt.axis("square")
    # Diagonal line for guiding the eye
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    lims = np.concatenate((xlim, ylim))
    lims = np.array([min(lims), max(lims)])
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_autoscale_on(False)
    plt.plot([-100, 100], [-100, 100], "--k")
    if n_series > 1:
        # colorbar
        cbar = plt.colorbar(sc)
        tick_locs = (np.arange(n_series) + 1.5) * (n_series - 1) / n_series
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels(np.arange(n_series) + 1)
    # title
    if title is not None:
        plt.title(title)
    plt.tight_layout()
