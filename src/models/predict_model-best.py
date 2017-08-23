import click
import csv
import numpy as np
import pandas as pd
import train_model
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor


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

#    features = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
#       'YearRemodAdd', 'MasVnrArea', 'ExterQual', 'BsmtQual', 'BsmtExposure',
#       'BsmtUnfSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'KitchenQual',
#       'FireplaceQu', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
#       'OpenPorchSF', 'ScreenPorch', 'MSZoning_C (all)',
#       'Neighborhood_Crawfor', 'Neighborhood_StoneBr', 'Condition1_Norm',
#       'Exterior1st_BrkFace', 'Functional_Typ', 'SaleType_New',
#       'SaleCondition_Abnorml', 'BsmtFinSF']

#    gbr = GradientBoostingRegressor(max_depth=4)
#    gbr.fit(X_train[features], y_train)
#    gbr_train = gbr.predict(X_train[features])
#
#  
#    min_max_scaler = MinMaxScaler()
#    X_train_minmax = min_max_scaler.fit_transform(X_train[features])
#    X_test_minmax = min_max_scaler.transform(X_test[features])
##
#    lasso = Lasso(alpha=0.01, max_iter=50000)
#    lasso.fit(X_train_minmax, y_train)
#    lasso_train = lasso.predict(X_train_minmax)
#
#    svr_rbf = SVR(kernel='linear', C=1000)
#    svr_rbf.fit(X_train_minmax, y_train)
#    svr_train = svr_rbf.predict(X_train_minmax)
#
#    knb = KNeighborsRegressor(n_neighbors=5)
#    knb.fit(X_train[features], y_train)
#    knn_train = knb.predict(X_train[features])
#
#    training = pd.DataFrame(X_train[features])
#    training['GBR'] = gbr_train
#    training['LASSO'] = lasso_train
#    #training['SVR'] = svr_train
#    training['KNN'] = knn_train
#
#
#    gbr_test = gbr.predict(X_test[features])
#    lasso_test = lasso.predict(X_test_minmax)
#    svr_test = svr_rbf.predict(X_test_minmax)
#    knn_test = knb.predict(X_test[features])
#
#    testing = pd.DataFrame(X_test[features])
#    testing['GBR'] = gbr_test
#    testing['LASSO'] = lasso_test
#    #testing['SVR'] = svr_test
#    testing['KNN'] = knn_test
#
#    mkn = KNeighborsRegressor(n_neighbors=50, weights='distance')
#    mkn.fit(training, y_train)

    
    pipe = train_model.build_fit_pipe(X_train, y_train)       
    prediction = pipe.predict(X_test)



#    prediction = []
#    for g,l in zip(gbr_prediction, lasso_prediction):
 #       prediction.append(0.2*l + 0.8*g)
        
        #0.3 0.7 - 1708-lasso02 - 0.12963
        #0.2 0.8 - 1708-lasso03 - 0.12916 - improvement by using lasso on scaled data  : 0.12858
        #0.1 0.9 - 1708-lasso04 - 0.13056
        
#
#    rfegbr = GradientBoostingRegressor()
#    rfe = RFE(estimator=rfegbr, n_features_to_select=10, step=1)    
#    gbr = GradientBoostingRegressor(max_depth=4, learning_rate=0.1)
#    pipe = Pipeline([("rfe", rfe), ("gbr", gbr)])
#    pipe.fit(X_train, y_train)
#    prediction = pipe.predict(X_test)

    
    # build, fit and predict
#    model = train_model.build_fit_model(X_train.drop('Id', axis=1), y_train)
#    prediction = model.predict(X_test.drop('Id', axis=1))
    
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
