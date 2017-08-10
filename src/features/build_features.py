import click
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest



def read_raw_data(input_filepath):
    """
    Reads CSV file from input_filepath and returns a Pandas dataframe
    """
    return pd.read_csv(input_filepath)    


def drop_columns(df, todrop):
    """
    Returns a view of df with columns in list todrop removed
    """
    return df.drop(todrop, axis=1)
    

def transform_ordinals(df, ordinals, mapping):        
    """
    Replaces values in df on columns in list ordinals following 
    mapping provided
    """
    return df[ordinals].replace(mapping)
    

def transform_categoricals(df):
    """
    Returns view of df where categorical features have been replaced by 
    multiple features using One Hot Encoding
    """
    df = pd.get_dummies(df)
    return df


def reduce_categoricals(df, categoricals):
    """
    Perform some binning operation to reduce the number of values 
    in categorical features provided
    """
    #TBC
    return
    

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--mode', type=click.Choice(['test', 'train']))
def main(input_filepath, output_filepath, mode):

    # read original data into data frame
    df = read_raw_data(input_filepath)
      
    # transform ordinal features represented as categories to numeric
    # this allows the model to reason about the distance between different values
    ordinals = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'HeatingQC',
               'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
    
    # First uniformize the set of string values
    valmap = {'Mn': 'Po', 'No': np.nan}     
    df[ordinals] = transform_ordinals(df, ordinals, valmap)
    
    # replace the NA values with an empty string so they can be used as key
    # account in the next mapping
    df[ordinals] = df[ordinals].fillna('')
    
    valmap = {'': 0, 'Po': 1, 'Fa': 2, 'Av': 3, 'TA': 3, 'Gd': 4, 'Ex': 5} 
    df[ordinals] = transform_ordinals(df, ordinals, valmap)
    df[ordinals] = df[ordinals].astype(int)

    # perform OneHotEncoding on categorical variables
    df['MSSubClass'] = df['MSSubClass'].astype(str)
    df = transform_categoricals(df)
        
    # correct the garage years
    df['GarageYrBlt'] = np.where((df['YearBuilt'] > df['GarageYrBlt']) & 
      (df['GarageYrBlt'] > 0), df['YearBuilt'], df['GarageYrBlt'])
    df['GarageYrBlt'] = df['GarageYrBlt'].astype(int)

    # default NAs to 0   
    df = df.fillna(0)

    if mode == 'train':
        # remove outliers for TRAINING DATA
        clf = IsolationForest(random_state=0)
        clf.fit(df)
        outlier = clf.predict(df)    
        df = df[outlier == 1]    

    # save the result of this processing as interim data
    df.to_csv(output_filepath, index=False)
    

    

if __name__ == '__main__':   
    main()
