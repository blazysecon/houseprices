# -*- coding: utf-8 -*-
import click
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib.backends.backend_pdf import PdfPages



def scatter_plot_by_group(groups, x, y, ax, title, xlabel, ylabel, diag=False):
    """ Adds grouped data to an existing axes instance resulting in colored scatter points.
        Color of data point depends on group they belong to.
    """
    for name, group in groups:
        h, = ax.plot(group[x], group[y], marker='o', linestyle='', ms=3, label=name)
        ax.legend()
        ax.set(adjustable='box-forced', aspect='equal')
        ax.set_xlabel(xlabel) 
        ax.set_ylabel(ylabel) 
        ax.set_title(title)
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
    if diag:
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    
        
def visualize_feature(x, y, ax, xlabel, ylabel, title, xnumeric):
    """ Adds data points to an existing axes instance resulting in a scatter plot.
        Can handle numeric and categorical data on the x axis, y axis is supposed numeric.
        Non-numeric data is always considered to be categorical.
    """
    if xnumeric == True:
        ax.scatter(x, y) 
        labels = ax.get_xticklabels() 
        for label in labels: 
            label.set_rotation(45) 
    else:     
        integer_map = dict([(val, i) for i, val in enumerate(set(x))]) 
        ax.scatter(x.apply(lambda z: integer_map[z]), y)
        fl = FixedLocator(range(len(integer_map)))
        ax.xaxis.set_major_locator(fl)
        ax.xaxis.set_ticklabels([str(k) for k in integer_map.keys()], rotation=45)
    ax.set_xlabel(xlabel) 
    ax.set_ylabel(ylabel) 
    ax.set_title(title)
             

def visualize_features(df, features, target, ncols, figsize, nplots_per_page, outpath):
    """ Plots the specified features against the target as scatter plots
    and adds them to a multicolumn (ncols) multi-page PDF document (nplots_per_page) 
    at location specified by outpath
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
                visualize_feature(df[f], df[target], ax, f, target, f, False)
            else:
                visualize_feature(df[f], df[target], ax, f, target, f, True)
        axlist = axes.flatten()             
        for unused in range(len(to_plot), len(axlist)):        
            axlist[unused].axis('off')           
        plt.tight_layout()
        pdf_pages.savefig(fig, dpi=10) 
        i = i + nplots_per_page
    pdf_pages.close()
    
        

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    df = pd.read_csv(input_filepath)
    target = 'SalePrice'
    features = df.columns
    visualize_features(df, features, target, 3, (15,15), 6, output_filepath)    
    
if __name__ == '__main__':
    main()