import numpy as np
import matplotlib.pyplot as plt

# this is also called a 'parity' plot
def plotyy(y, y_pred, color=None, title=None):
    # Customize colors to match with capacity vs cycle plot earlier
    if color is not None:
        n_series = np.unique(color).size
    else:
        n_series = 1
    cmap = plt.get_cmap('tab10', n_series)
    # Get axes and plot
    fig, ax = plt.subplots(1,1, figsize=(4,3))
    sc = ax.scatter(y.values, y_pred, s=8, c=color, cmap=cmap, marker='.')
    ax.set_aspect('equal')
    plt.xlabel('Actual relative\ndischarge capacity')
    plt.ylabel('Predicted relative\ndischarge capacity')
    plt.axis('square')
    # Diagonal line for guiding the eye
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    lims = np.concatenate((xlim, ylim))
    lims = np.array([min(lims), max(lims)])
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_autoscale_on(False)
    plt.plot(lims, lims, '--k')
    # if n_series > 1:
    #     # colorbar
    #     cbar = plt.colorbar(sc)
    #     tick_locs = (np.arange(n_series) + 1.5)*(n_series-1)/n_series
    #     cbar.set_ticks(tick_locs)
    #     cbar.set_ticklabels(np.arange(n_series)+1)
    # title
    if title is not None:
        plt.title(title, fontsize=10)

    ax.set_xticks(ax.get_yticks())
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    plt.tight_layout()

def plot_parity_bin(y_true, y_pred):
    fig, ax = plt.subplots(1,1, figsize=(4,3))
    # h = ax.hist2d(y, y_pred_xgb_2freq, bins=25, cmap='Blues_white0')
    h = ax.hexbin(y_true, y_pred, gridsize=20, cmap='Blues_white0')
    ax.set_aspect('equal')
    plt.xlabel('Actual relative\ndischarge capacity')
    plt.ylabel('Predicted relative\ndischarge capacity')
    plt.axis('square')
    # plt.colorbar(label='Counts')
    # Diagonal line for guiding the eye
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    lims = np.concatenate((xlim, ylim))
    lims = np.array([min(lims), max(lims)])
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_autoscale_on(False)
    plt.plot(lims, lims, '--k')