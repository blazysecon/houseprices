import click
import pandas as pd
import numpy as np


def read_raw_data(input_filepath):
    return pd.read_csv(input_filepath)    

def drop_columns(df, todrop):
    return df.drop(todrop, axis=1)
    
def transform_ordinals(df, ordinals, mapping):        
   return df[ordinals].replace(mapping)
    
def transform_categoricals(df):
    df = pd.get_dummies(df)
    df = df.fillna(0)
    return df

def reduce_categoricals(df, categoricals):
    #do something
    print()
    
@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):

    # read original data into data frame
    df = read_raw_data(input_filepath)

    # drop ID column as it doesn't add any information    
    todrop = ['Id']
    df = drop_columns(df, todrop)
    
    
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
    
    # default NAs to 0   
    df = df.fillna(0)

    # save the result of this processing as interim data
    df.to_csv(output_filepath)
    

    

if __name__ == '__main__':   
    main()
