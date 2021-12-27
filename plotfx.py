import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

def single_plot_xtime(df, metric, title, ylabel):
    """
    Generates a single plot of results over time; includes separate grey
    lines for each run and a single blue line for the average of all runs
    -------
    Inputs
    -------
    df: pandas DataFrame
        Dataframe to use for plotting
    metric: string
        Name of dataframe column to use for plotting
    title: string
        Plot title
    ylabel: string
        Y-axis label
    """
    mpl.rcParams['axes.titlesize'] = 'x-large'
    mpl.rcParams['axes.labelsize'] = 'x-large'

    for i in range(1, df['rep'].max() + 1):
        plt.plot(df[df.rep == i]['tstep'].values, df[df.rep == i][metric].values, 'lightgrey', linewidth = 0.75)
    plt.plot(df.groupby('tstep')[metric].mean().index, df.groupby('tstep')[metric].mean().values)
    plt.title(title)
    plt.xlabel('Time (Days)')
    plt.ylabel(ylabel)
    plt.show();

def row_plot_xtime_bytest(frames, metric, ylabel, titles, suptitle = None):
    """
    Generates a row of plots for a single metric over time; each plot represents
    a different dataframe; includes separate grey lines for
    each run and a single blue line for the average of all runs
    -------
    Inputs
    -------
    frames: list of pandas DataFrame
        Dataframes to use for plotting, each will be plotted in a separate
        column
    metric: string
        Name of dataframe column to use for plotting
    ylabel: string
        Y-axis label
    titles: list of strings
        Plot titles; one for each dataframe/column
    suptitle: string, optional
        Main title
    """
    mpl.rcParams['axes.titlesize'] = 'x-large'
    mpl.rcParams['axes.labelsize'] = 'x-large'

    f, axs = plt.subplots(1, len(frames), figsize = (20, 5), sharey = True)
    f.suptitle(suptitle, fontsize = 15)
    f.subplots_adjust(wspace=0, hspace=0)

    for (i,df) in enumerate(frames):
        for j in range(1, df['rep'].max() + 1):
            axs[i].plot(df[df.rep == j]['tstep'].values, df[df.rep == j][metric].values, 'lightgrey', linewidth = 0.75)
        axs[i].plot(df.groupby('tstep')[metric].mean().index, df.groupby('tstep')[metric].mean().values)
        axs[i].set_title(titles[i])
        axs[i].set_xlabel('Time (Days)')
        if i==0:
            axs[i].set_ylabel(ylabel)

def grid_plot_xtime_bytest(frame, row_metric, col_metric, plot_metric, ylabel, titles, suptitle = None):
    """
    Generates a grid of plots over time; each row and column represents a
    different metric; includes separate grey lines for each run and a single
    blue line for the average of all runs
    -------
    Inputs
    -------
    frame: pandas DataFrame
        Dataframe to use for plotting
    row_metric: string
        Name of dataframe column to use for the row; each unique value will be
        plotted in a separate row
    col_metric: string
        Name of dataframe column to use for the column; each unique value will
        be plotted in a separate column
    plot_metric: string
        Name of dataframe column to use for plotting
    ylabel: string
        Y-axis label
    titles: list of strings
        Plot titles; one for each column
    suptitle: string, optional
        Main title
    """
    mpl.rcParams['axes.titlesize'] = 'x-large'
    mpl.rcParams['axes.labelsize'] = 'x-large'

    nrows = frame[row_metric].nunique()
    ncols = frame[col_metric].nunique()

    f, axs = plt.subplots(nrows, ncols, figsize = (20, 15), sharex = True, sharey = True)
    f.suptitle(suptitle, fontsize = 20, y = 0.93)
    f.subplots_adjust(wspace=0, hspace=0)

    # Loop through the rows
    for (i, row) in enumerate(frame[row_metric].unique()):
        # Loop through the columns
        for (j, col) in enumerate(frame[col_metric].unique()):
            subdat = frame.loc[(frame[row_metric]==row) & (frame[col_metric]==col)]
            for k in range(1, subdat['rep'].max() + 1):
                axs[i,j].plot(subdat[subdat.rep == k]['tstep'].values, subdat[subdat.rep == k][plot_metric].values, 'lightgrey', linewidth = 0.75)
            axs[i,j].plot(subdat.groupby('tstep')[plot_metric].mean().index, subdat.groupby('tstep')[plot_metric].mean().values)
            # Only set title if first row
            if i==0:
                axs[i,j].set_title(titles[j])
            # Only set xlabel if last row
            if i==(nrows-1):
                axs[i,j].set_xlabel('Time (Days)')
            # Only set ylabel if first column
            if j==0:
                axs[i,j].set_ylabel(ylabel)
