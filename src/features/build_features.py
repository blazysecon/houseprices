import click
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import FunctionTransformer


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
    

def transform_ordinals(df):        
    """
    Replaces values in df on columns in list ordinals following 
    mapping provided
    """
    # transform ordinal features represented as categories to numeric
    # this allows the model to reason about the distance between different values
    ordinals = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'HeatingQC',
               'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
    
    # First uniformize the set of string values
    valmap = {'Mn': 'Po', 'No': np.nan}     
    df[ordinals] = df[ordinals].replace(valmap)
    
    # replace the NA values with an empty string so they can be used as key
    # account in the next mapping
    df[ordinals] = df[ordinals].fillna('')
    
    valmap = {'': 0, 'Po': 1, 'Fa': 2, 'Av': 3, 'TA': 3, 'Gd': 4, 'Ex': 5} 
    df[ordinals] = df[ordinals].replace(valmap)
    df[ordinals] = df[ordinals].astype(int)

    return df
    

def transform_categoricals(df):
    """
    Returns view of df where categorical features have been replaced by 
    multiple features using One Hot Encoding
    """
    df['MSSubClass'] = df['MSSubClass'].astype(str)
    oldcolumns = df.columns
    categoricals = df.select_dtypes(include=['object', 'category']).columns    
    df = pd.get_dummies(df)
    cat_map = {}
    newcolumns = list(set(df.columns) - set(oldcolumns))
    for cat in categoricals:
        categories = filter(lambda x: x.startswith(str(cat+"_")), newcolumns)
        cat_map[cat] = list(categories)
    return df, cat_map


def correct_garageyrblt(df):
    # default NAs to 0   
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)

    # correct the garage years
    df['GarageYrBlt'] = np.where((df['YearBuilt'] > df['GarageYrBlt']) & 
      (df['GarageYrBlt'] > 0), df['YearBuilt'], df['GarageYrBlt'])
    df['GarageYrBlt'] = df['GarageYrBlt'].astype(int)
    return df

def aggregate_bsmt(df):
    df['BsmtFinSF'] = df['BsmtFinSF1'] + df['BsmtFinSF2']
    df = df.drop(['TotalBsmtSF', 'BsmtFinSF1', 'BsmtFinSF2'], axis=1)
    return df    

def revert_to_original(transformed_df, categorical_map, original_columns):
    #https://tomaugspurger.github.io/categorical-pipelines.html
    series = []
    categorical = [k for k in categorical_map.keys()]
    non_categorical = list(set(original_columns) - set(categorical))    
    for col, dums in categorical_map.items():    
        code_dict = {k:v for (v,k) in enumerate(dums)}        
        categories = transformed_df[dums].idxmax(axis=1)
        codes = [code_dict[k] for k in categories]
        cats = pd.Categorical.from_codes(codes, [d[len(col)+1:len(d)] for d in dums])
        series.append(pd.Series(cats, name=col))    
    cat_df = pd.DataFrame(series).T
    df = pd.concat([transformed_df[non_categorical], cat_df], axis=1)        
    df = df[original_columns]
    return df        
        
def reduce_categoricals(df, categoricals):
    """
    Perform some binning operation to reduce the number of values 
    in categorical features provided
    """
    #TBC
    return
    
def log_of_target(X):
    return np.log1p(X['SalePrice'])

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--mode', type=click.Choice(['test', 'train', 'eval']))
def main(input_filepath, output_filepath, mode):

    # read original data into data frame
    df = read_raw_data(input_filepath)
      
    #original_columns = df.columns
        
    # represent ordinal features as numeric feature to allow
    # ordering between them    
    df = transform_ordinals(df)

    # perform OneHotEncoding on categorical variables
    df, cat_map = transform_categoricals(df)
    
    df = aggregate_bsmt(df)
        
    # correct the garage years
    df = correct_garageyrblt(df)

    # default NAs to 0   
    df = df.fillna(0)

    if mode == 'train':
        # remove outliers for TRAINING DATA
        clf = IsolationForest(random_state=0)
        clf.fit(df)
        outlier = clf.predict(df)    
        df = df[outlier == 1]    
        
#    if mode == 'train':
#        df['SalePrice'] = np.log1p(df['SalePrice'])
#    df['LotArea'] = np.log1p(df['LotArea'])
#    df['LotFrontage'] = np.log1p(df['LotFrontage'])
# 
    # save the result of this processing as interim data
    df.to_csv(output_filepath, index=False)

#    original = revert_to_original(df, cat_map, original_columns)
    
#    original.to_csv(output_filepath, index=False)
        

if __name__ == '__main__':   
    main()
