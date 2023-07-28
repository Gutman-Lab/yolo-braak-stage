# Functions:
# - plot_bars
# - format_plot_edges
# - plot_histogram
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from pandas import DataFrame
import numpy as np
from typing import Union, List, Tuple


def plot_bars(
    df: DataFrame, x_col: str, y_col: str, group: str = None, title: str = None,
    title_fs: int = 24, fs: int = 22, figsize: (int, int) = (7,5), 
    save_fp: str = None, x_label: str = '', y_label: str = None, 
    y_lim: (int, int) = None, show_plot: bool = False, 
    x_tick_rotation: int = 45, order: list = None, xtick_fs = 20, ytick_fs = 16,
    hide_ylabel: bool = False, hide_xlabel: bool = False, ha: str = 'center',
    **kwargs: dict
) -> matplotlib.axes:
    """
    Create bar plots from a dataframe containing the data. Allows grouping 
    different rows into single columns using the group parameter.
    
    Args:
        df: Data to plot.
        x_col: Column in df parameter for the bar groups.
        y_col: Column in df parameter for the height of the bars.
        group: Used for grouped bar plots, column in df parameter to group by.
        title: Title of figure.
        title_fs: Size of figure title.
        fs: Font size of non-title font.
        figsize: Size of figure (width, height).
        save_fp: File path to save figure to.
        x_label: Label to use on the x-axis.
        y_label: Label for the y axis.
        y_lim : Range of the y-axis. Set to None to be inferred from data.
        show_plot: If True then the plot will be shown in the output, for 
            Jupyter notebook / lab use.
        x_tick_rotation: Rotation of x-tick labels.
        order: Bars to plot and in the order to plot them.
        xtick_fs: Size of xtick labels.
        ytick_fs: Size of ytick labels.
        hide_ylabel: Hide y label.
        hide_xlabel: Hide x label.
        ha: Alignment of xticks: 'center', 'left', 'right'.
        kwargs: Other keyword arguments are passed through to 
            matplotlib.axes.Axes.bar() & seaborn.barplot(). Do not pass the 
            keyword argument data, x, or y.
        
    Returns:
        Matplotlib ax.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=df, x=x_col, y=y_col, hue=group, order=order, zorder=5, **kwargs)
    
    # get the max value of the bar
#     avgs = []

#     for x in df[x_col].unique():
#         avgs.append(df[df[x_col] == x][y_col].mean())

#     max_bar = max(avgs)
    
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.yticks(fontweight='bold', fontsize=ytick_fs)
    plt.xticks(fontweight='bold', fontsize=xtick_fs, rotation=x_tick_rotation, ha='center')
    
    if hide_ylabel:
        plt.ylabel(None)
    elif y_label is not None:
        plt.ylabel(y_label, fontsize=fs, fontweight='bold')
    else:
        plt.ylabel(y_col, fontsize=fs, fontweight='bold')
        
    if hide_xlabel:
        plt.xlabel(None)
    elif x_label is not None:
        plt.xlabel(x_label, fontsize=fs, fontweight='bold')
    else:
        plt.xlabel(x_col, fontsize=fs, fontweight='bold')
        
    ax.tick_params(axis='both', which='both', direction='out', length=10, width=3)
#     plt.axhline(max_bar, color='k', linestyle='dashed', linewidth=2)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
        
    if title is not None:
        plt.title(title, fontsize=fs, fontweight='bold')
        
    return ax


def format_plot_edges(ax: matplotlib.axes.Axes, lw: int = 3):
    """Format plot edges for publication by removing the top and right edges
    and changing the width of the left and bottom.
    
    Args:
        ax: Plot axis.
        lw: Line width of edges.
        
    Returns:
        ax: The formatted axes.
        
    """
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(lw)
    ax.spines['left'].set_linewidth(lw)
    
    return ax


def plot_histogram(values: Union[List[int], List[float]], spine_lw: int = 3, 
                   fs: int=18, x_label: str = None, title: str = None, 
                   title_fs: int = 18, y_freq: Union[int, float] = None, 
                   figsize: Tuple[int, int] = (4,4), **kwargs: dict):
    """Plot a histogram.
    
    Args:
        values: List of values to plot in histogram.
        spine_lw: Line width of plot edges or spines.
        fs: Font-size of ticks and axes labels.
        x_label: Label on the x-axis.
        title: Title on plot.
        title_fs: Title fontsize.
        y_freq: The frequency used for the yticks.
        **kwargs: Key-word arguments to pass to matplotlib hist function.
        
    """
    plt.figure(figsize=figsize)
    ax = sns.histplot(values, **kwargs)
    ax = format_plot_edges(ax, lw=spine_lw)
    
    # Make ticks nicer.
    ax.tick_params(axis='both', which='both', direction='out', length=10, 
                   width=spine_lw)
    
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.ylabel('Frequency', fontsize=fs)
    plt.xlabel(x_label, fontsize=fs)
    plt.title(title, fontsize=title_fs)
    
    # Adjust the frequency of the y-axis ticks.
    if y_freq is not None:
        y_max = []
        
        for p in ax.patches:
            y_max.append(p.get_height())
            
        y_pad = 2 if isinstance(y_freq, int) else 0.2
        
        plt.yticks(np.arange(0, max(y_max) + y_pad, y_freq))
    
    return ax
