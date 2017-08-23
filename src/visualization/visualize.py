# -*- coding: utf-8 -*-
import click
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib.backends.backend_pdf import PdfPages

def classify_score(score):
    if score <= 0.1:
        return 0
    elif score <= 0.15:
        return 2
    elif score <= 0.2:
        return 3
    else:
        return 4

def transform_scores(scores):
    transformed =  list(map(lambda x: classify_score(x), scores))    
    return transformed


def scatter_plot_by_group(groups, x, y, ax, title, xlabel, ylabel, diag=False):
    """ Adds grouped data to an existing axes instance resulting in colored scatter points.
        Color of data point depends on group they belong to.
        
    :param groups: grouped data
    :param x: name of column to use as x-coordinates
    :param y: name of column to use as y-coordinates
    :param ax: axes object on which to add plot
    :param title: title of plot
    :param xlabel: label of x-axis
    :param ylabel: label of y-axis
    :param diag: indicates whether a diagonal should be added to the plot or not
    :type groups: groupby object
    :type x, y: string
    :type title, xlabel, ylabel: string
    :return: void
    """
    for name, group in groups:
        h, = ax.plot(group[x], group[y], marker='o', linestyle='', ms=3, label=name)
        ax.legend()
        #ax.set(adjustable='box-forced', aspect='equal')
        ax.set_xlabel(xlabel) 
        ax.set_ylabel(ylabel) 
        ax.set_title(title)
        #ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
    if diag:
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    
        
def visualize_feature(x, y, color, cmap, ax, xlabel, ylabel, title, xnumeric):
    """ Adds data points to an existing axes instance resulting in a scatter plot.
        Can handle numeric and categorical data on the x axis, y axis is supposed numeric.
        Non-numeric data is always considered to be categorical.
        
    :param x: x-coordinates
    :param y: y-coordinates
    :param color: colors of cmap to use
    :param cmap: color map
    :param ax: axes object on which to add plot
    :param title: title of plot
    :param xlabel: label of x-axis
    :param ylabel: label of y-axis
    :param numeric: indicates whether x-axis data is numeric or categorical
    :type x, y: lists
    :type title, xlabel, ylabel, cmap: string
    :type color: list of ints
    :return: void
    """
    if xnumeric == True:        
        ax.scatter(x, y, c=color, cmap=cmap) 
        labels = ax.get_xticklabels() 
        for label in labels: 
            label.set_rotation(45)  
    else:     
        integer_map = dict([(val, i) for i, val in enumerate(set(x))]) 
        ax.scatter(x.apply(lambda z: integer_map[z]), y, c=color, cmap=cmap)
        fl = FixedLocator(range(len(integer_map)))
        ax.xaxis.set_major_locator(fl)
        ax.xaxis.set_ticklabels([str(k) for k in integer_map.keys()], rotation=45)
    ax.set_xlabel(xlabel) 
    ax.set_ylabel(ylabel) 
    ax.set_title(title)
             

def visualize_features(df, features, target, color, cmap, ncols, figsize, nplots_per_page, outpath):
    """ Plots the specified features against the target as scatter plots
    and adds them to a multicolumn (ncols) multi-page PDF document (nplots_per_page) 
    at location specified by outpath
    
    :param df: input data
    :param features: names of feature columns
    :param target: name of target feature
    :param color: list used to determine colors using cmap
    :param cmap: color map to use
    :param ncols: number of individual plots per row
    :param figsize: size of individual plots
    :param nplots_per_page: number of individual plots to put on a page
    :param outpath: path to output file
    :type df: pandas dataframe
    :type features: list of strings
    :type target, cmap: string
    :type color: list of ints
    :type ncols, nplots_per_page: int
    :type figsize: tuple (int, int)
    :type outpath: string
    :return: void
    """
    nplots = len(features)
    nrows = int(np.ceil(nplots_per_page / float(ncols)))
    npages = int(np.ceil(nplots / float(nplots_per_page)))

    pdf_pages = PdfPages(outpath)
    i = 0
    for pg in range(npages):
        to_plot = features[i:i+nplots_per_page]
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=10)
        for f, ax in zip(to_plot, axes.flatten()):
            if(df[f].dtype == np.object):
                visualize_feature(df[f], df[target], color, cmap, ax, f, target, f, False)
            else:
                visualize_feature(df[f], df[target], color, cmap, ax, f, target, f, True)
        axlist = axes.flatten()             
        for unused in range(len(to_plot), len(axlist)):        
            axlist[unused].axis('off')           
        plt.tight_layout()
        pdf_pages.savefig(fig, dpi=10) 
        i = i + nplots_per_page
        plt.close()
    pdf_pages.close()
    

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--mode', type=click.Choice(['features', 'score']))

def main(input_filepath, output_filepath, mode):
    df = pd.read_csv(input_filepath)
    target = 'SalePrice'
    features = df.columns

    if mode == 'features':
        colors = [1 for i in range(0, len(df))]
        visualize_features(df, features, target, colors, 'viridis', 3, (15,15), 6, output_filepath)  
    if mode == 'score':    
        features = features[1:len(features)] 
        ts = transform_scores(df['Score'])
        visualize_features(df, features, target, ts, 'autumn_r', 3, (15,15), 6, output_filepath)
    
if __name__ == '__main__':
    main()