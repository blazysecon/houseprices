import click
import pandas as pd
import train_model

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score


    
@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
def main(input_filepath):

    # read the processed data
    df = pd.read_csv(input_filepath)

    # remove target
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice'] 
    
    # split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    model = train_model.build_fit_model(X_train, y_train)
    print("Training set score : {:.3f}".format(model.score(X_train, y_train)))
    print("Test set score : {:.3f}".format(model.score(X_test, y_test)))

    # perform cross validation to get a more reliable score for the model
    shuffle_split = ShuffleSplit(test_size=.5, train_size=.5, n_splits=10)
    scores = cross_val_score(model, X, y, cv=shuffle_split)
    print("Cross val scores : \n{}".format(scores))
    print("Mean Cross val score : \n{:.2f}".format(scores.mean()))
    


if __name__ == '__main__':   
    main()
