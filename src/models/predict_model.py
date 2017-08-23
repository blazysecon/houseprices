import click
import csv
import numpy as np
import pandas as pd
import train_model

@click.command()
@click.argument('train_filepath', type=click.Path(exists=True))
@click.argument('test_filepath', type=click.Path(exists=True))
@click.argument('predict_filepath', type=click.Path())
def main(train_filepath, test_filepath, predict_filepath):
    """
    Reads training data and trains model from train_model.py on it. Then reads test data
    and uses model to make prediction for test data. This prediction is then written to the output file.
    """
    # read the prepared data
    X_train = pd.read_csv(train_filepath)
    X_test = pd.read_csv(test_filepath)  
        
    # split training data in features and target
    y_train = X_train['SalePrice']
    X_train =  X_train.drop(['SalePrice'], axis=1)

    # make sure test and training dataset have the same columns
    col_to_add = np.setdiff1d(X_train.columns, X_test.columns)
    col_to_rem = np.setdiff1d(X_test.drop('Id', axis=1).columns, X_train.columns)
    for c in col_to_add:
        X_test[c] = 0
    X_test = X_test.drop(col_to_rem, axis=1)

    # make sure column order is the same for training and test dataset
    X_test = X_test[X_train.columns]
  
    pipe = train_model.build_fit_pipe(X_train.drop('Id', axis=1), y_train)       
    prediction = pipe.predict(X_test.drop('Id', axis=1))
    
    # write prediction to output file
    result = zip(X_test['Id'], prediction)            
    with open(predict_filepath, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["Id", "SalePrice"])
        for Id, pred in result:
            writer.writerow([Id, pred])
            print("ID: {}, P: {}".format(Id, pred))
            

if __name__ == '__main__':   
    main()
