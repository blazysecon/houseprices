import click
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score


def read_raw_data(input_filepath):
    return pd.read_csv(input_filepath)    

def drop_columns(df, todrop):
    df = df.drop(todrop, axis=1)
    
def transform_ordinals(df, ordinals, mapping):        
    df[ordinals] = df[ordinals].replace(mapping)
    
def transform_categoricals(df, categoricals):
    df = df.get_dummies(df)
    df = df.fillna(0)

def reduce_categoricals(df, categoricals):
    #do something
    print()
    
@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
def main(input_filepath):

    # read the processed data
    df = pd.read_csv(input_filepath)
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']

    # split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    # initialize and fit a Gradient Boosting Decision Tree model
    gbr = GradientBoostingRegressor(max_depth=4, learning_rate=0.1)
    gbr.fit(X_train, y_train)
    print("Training set score : {:.3f}".format(gbr.score(X_train, y_train)))
    print("Test set score : {:.3f}".format(gbr.score(X_test, y_test)))

    # perform cross validation to get a more reliable score for the model
    shuffle_split = ShuffleSplit(test_size=.5, train_size=.5, n_splits=10)
    scores = cross_val_score(gbr, X, y, cv=shuffle_split)
    print("Cross val scores : \n{}".format(scores))
    print("Mean Cross val score : \n{:.2f}".format(scores.mean()))
    

if __name__ == '__main__':   
    main()
