import click
import pandas as pd
import train_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from math import sqrt

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', required=False)
@click.argument('original_filepath', required=False)
def main(input_filepath, output_filepath=None, original_filepath=None):
    """
    Reads input data, splits it into training and test data and makes prediction
    using model from train_model.py
    """
    
    # read the processed data
    df = pd.read_csv(input_filepath)

    # remove target
    X = df.drop(['SalePrice'], axis=1)
    y = df['SalePrice'] 
    
    # split in training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    pipe = train_model.build_fit_pipe(X_train.drop('Id', axis=1), y_train)       
    pred = pipe.predict(X_test.drop('Id', axis=1))
    total = 0
    score = []
    for t,p in zip(y_test, pred):
        error = sqrt(mean_squared_log_error([t],[p])) 
        score.append(error)
        print("truth : {}, prediction : {}, score: {} ".format(t, p, error))
        total = total + error
    print(total/float(len(y_test)))
    
    if (output_filepath != None) & (original_filepath != None):
        original = pd.read_csv(original_filepath)
        result = pd.DataFrame(X_test)
        result = result[['Id']]
        result = pd.merge(result, original, on='Id', how='inner')
        result['Prediction'] = pred
        result['Score'] = score
        result.to_csv(output_filepath)

if __name__ == '__main__':   
    main()
