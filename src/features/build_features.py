import click
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


def read_raw_data(input_filepath):
    """
    Reads CSV file from input_filepath and returns a Pandas dataframe
    
    :param input_filepath: path to csv file containing data to be read
    :type input_filepath: string
    :return: dataframe containing data from input_filepath
    :rtype: pandas dataframe
    """
    return pd.read_csv(input_filepath)    


def drop_columns(df, todrop):
    """
    Returns a view of df with columns in list todrop removed
    
    :param df: dataframe from which a subset of columns should be dropped
    :type df: pandas dataframe 
    :param todrop: list of column names to be dropped from df
    :type df: list of strings
    :return: view of the dataframe wihtout the columns listed in todrop
    :rtype: pandas dataframe    
    """
    return df.drop(todrop, axis=1)
    

def transform_ordinals(df):        
    """
    Returns view of df with nominal encodings of the ordinal features 
    transformed to numeric encoding (in order to allow ordering).
    The list of features to be modified is hardcoded in the function.
    
    :param df: dataframe on which the encoding of ordinal features should be modified
    :type df: pandas dataframe 
    :return: view of dataframe where ordinal features have been encoded numerically
    :rtype: pandas dataframe    
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
    Returns tuple (v, catmap). Where v is view of df with categorical features 
    having been replaced by multiple boolean features using One Hot Encoding; 
    catmap is a dictionary mapping original feature to features it has been 
    replaced with.    
    
    :param df: dataframe on which the encoding of categorical features should be modified
    :type df: pandas dataframe 
    :return: view of dataframe where categorical features have been one hot encoded
    :rtype: pandas dataframe       
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
    """
    Returns view of df where the year in which garage was built 
    (feature GarageYrBlt), if smaller than year in which house was built 
    (feature YearBuilt), is defaulted to the latter value
    
    :param df: dataframe on which value of the feature 'GarageYrBlt' should be corrected
    :type df: pandas dataframe 
    :return: view of dataframe where concerned values have been corrected
    :rtype: pandas dataframe       
    """
    # default NAs to 0   
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)

    # correct the garage years
    df['GarageYrBlt'] = np.where((df['YearBuilt'] > df['GarageYrBlt']) & 
      (df['GarageYrBlt'] > 0), df['YearBuilt'], df['GarageYrBlt'])
    df['GarageYrBlt'] = df['GarageYrBlt'].astype(int)
    
    return df


def aggregate_bsmt(df):
    """
    Returns view of df where feature TotalBsmtSF, BsmtFinSF1, BsmtFinSF2 have 
    been replace by sole feature BsmtFinSF
    
    :param df: dataframe on which columns relating to finished basement surface have been merged
    :type df: pandas dataframe 
    :return: view of dataframe where original columns have been replaced
    :rtype: pandas dataframe       
    """
    df['BsmtFinSF'] = df['BsmtFinSF1'] + df['BsmtFinSF2']
    df = df.drop(['TotalBsmtSF', 'BsmtFinSF1', 'BsmtFinSF2'], axis=1)
    
    return df    


def revert_to_original(transformed_df, categorical_map, original_features):
    """
    Utility function allowing to revert a dataframe where categoricals have been
    transformed to dummy features to its original form with categoricals.
    
    :param transformed_df: dataframe which has been transformed using one hot encoding
    :param categorical_map: dictionary mapping original categorical features to dummy features
    :param original_features: list of column names of the original dataframe before transformation
    :type transformed_df: pandas dataframe 
    :type categorical_map: dictionary key=original feature, value=list of corresponding dummy features 
    :type original_features: list of strings
    :return: view of transformed_df where the dummy features encoding a category have been replaced with the original categorical feature
    :rtype: pandas dataframe 

    note:: inspired by https://tomaugspurger.github.io/categorical-pipelines.html       
    """    
    
    series = []
    categorical = [k for k in categorical_map.keys()]
    non_categorical = list(set(original_features) - set(categorical))    
    for col, dums in categorical_map.items():    
        code_dict = {k:v for (v,k) in enumerate(dums)}        
        categories = transformed_df[dums].idxmax(axis=1)
        codes = [code_dict[k] for k in categories]
        cats = pd.Categorical.from_codes(codes, [d[len(col)+1:len(d)] for d in dums])
        series.append(pd.Series(cats, name=col))    
    cat_df = pd.DataFrame(series).T
    df = pd.concat([transformed_df[non_categorical], cat_df], axis=1)        
    df = df[original_features]
    
    return df        
        

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--mode', type=click.Choice(['test', 'train', 'eval']))
def main(input_filepath, output_filepath, mode):

    # read original data into data frame
    df = read_raw_data(input_filepath)      
         
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
        
    # save the result of this processing as interim data
    df.to_csv(output_filepath, index=False)
      

if __name__ == '__main__':   
    main()
