import numpy as np


def add_significance(ax, bar, q0=1, q1=2, xoff=0.1, yoff=0, text='*', fontsize=20, toff=0):
    xy = [(bar.lines[i].get_xydata()[0, 0], bar.lines[i].get_xydata()[-1, -1]) for i in range(0, len(bar.lines), 3)]
    xy = np.array(xy)
    ylim = bar.get_ylim()
    height = ylim[1]
    height = max(xy[q0, 1], xy[q1, 1])
    h=(ylim[1]-ylim[0])/30
    height+=h + yoff
    ax.plot([xy[q0, 0]+xoff, xy[q0, 0]+xoff, xy[q1, 0]-xoff, xy[q1, 0]-xoff], [height, height+h, height+h, height], lw=1.5, color=[0.54, 0.54, 0.54])
    ax.text((xy[q0, 0]+xy[q1, 0])/2, height+toff, text, fontsize=fontsize, horizontalalignment='center')


def filter_data(data, percentile=3):
    """Filter data by removing the lower and upper percentiles."""
    upper = np.percentile(data, 100 - percentile)
    lower = np.percentile(data, percentile)
    return data[(data > lower) & (data < upper)]


def extract_population_stats(wells, all_data, ch=0, sites=12):
    data_list = []
    for ad in all_data:
        st = [[np.median(ad[f'{w}-Site_{i}'][..., ch]) for i in range(sites)] for w in wells]
        data_list.append([val for sublist in st for val in sublist])
    return np.array(data_list)


def compute_median(data, min_val=0):
    return [np.median(i - min_val) for i in data]