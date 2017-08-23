from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
import itertools as it
import numpy as np
import pandas as pd

class StackingTransformer(BaseEstimator, TransformerMixin):
    """
    Custom Transformer that fits up to 4 different models and transforms input data
    into 4 different predictions. 
    
    The 4 different models are : 
        - Gradient Boosting Regressor
        - Lasso Regression
        - Support Vector Regressor
        - K-nearest Neighbors Regressor
        
    The models to use can be specified using boolean parameters.
    Caution : The parameters of the models have not been optimized, but have been set by hand. 
    Some of them can be specified using the transformer's parameters.
    """
    
    def __init__(self, dogbr=True, dolasso=False, dosvr=False, doknn=False, gbr_maxdepth=4, lasso_max_iter=50000, svr_kernel='linear', svr_C=1000, kn_n=20):
        """
        Initializes the transformer object. The models are not yet initialized, this is done upon calling the fit method.
        
        :param dogbr: indicates whether gradient bossting regression should be used (True) or not (False)
        :param dolasso: indicates whether Lasso regression should be used (True) or not (False)
        :param dosvr: indicates whether support vector regression model should be used (True) or not (False)
        :param doknn: indicates whether k-nearest neighbors regression should be used (True) or not (False)
        :param gbr_maxdepth: maximum depth of gradient boosting regressor trees
        :param lasso_max_iter: maximum number of iterations for lasso regression
        :param svr_kernel: type of kernel to use with the support vector regression
        :param svr_C: value of penalty parameter C of the svm
        :param kn_n: number of neighbors (k) in the k-nearerst neighbors regression
        :type dogbr: boolean
        :type dolasso: boolean
        :type dosvr: boolean
        :type doknn: boolean
        :type gbr_maxdepth: int
        :type lasso_max_iter: int
        :type svr_kernel: string ('linear', 'polynomial', 'rbf')
        :type svr_C: int
        :type kn_n: int
        :return: void
        """
    
        self.gbr_maxdepth = gbr_maxdepth
        self.lasso_max_iter = lasso_max_iter
        self.svr_kernel = svr_kernel
        self.svr_C = svr_C
        self.kn_n = kn_n
        self.dogbr = dogbr
        self.dolasso = dolasso
        self.dosvr = dosvr
        self.doknn = doknn


    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"dogbr": self.dogbr, 
                "dolasso": self.dolasso,
                "dosvr": self.dosvr,
                "doknn": self.doknn,
                "gbr_maxdepth" : self.gbr_maxdepth,
                "lasso_max_iter" : self.lasso_max_iter,
                "svr_kernel" : self.svr_kernel,
                "svr_C" : self.svr_C,
                "kn_n" : self.kn_n
                }


    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.__setattr__(parameter, value)
        return self



    def fit(self, X, y):
        """
        Initializes the models and fits them to the data.
        Gradient Boosting Regressor and K-nearest Neighbors Regressor are fit on unscaled data.
        Lasso and Support Vector Regressor are fit on log-transformed y and scaled X.
        X is scaled using a MinMaxScaler
        
        :param X: feature data on which transformer is to be fit
        :param y: target data on which transformer is to be fit
        :return: self
        """
        
        if self.dogbr:
            self.gbr_ = GradientBoostingRegressor(max_depth=self.gbr_maxdepth)
        if self.dolasso:    
            self.lasso_ = Lasso(max_iter=self.lasso_max_iter, alpha=0.01)
        if self.dosvr:    
            self.svr_ = SVR(kernel=self.svr_kernel, C=self.svr_C)
        if self.doknn:    
            self.kn_ = KNeighborsRegressor(n_neighbors=self.kn_n, weights='distance')
        
        if self.dosvr | self.dolasso:
            y_log = np.log1p(y)
            self.scaler_ = MinMaxScaler()        
            X_scaled = self.scaler_.fit_transform(X)
        
        if self.dogbr:
            self.gbr_.fit(X, y)
        if self.dolasso:            
            self.lasso_.fit(X_scaled, y_log)
        if self.dosvr:    
            self.svr_.fit(X_scaled, y_log)
        if self.doknn:            
            self.kn_.fit(X, y)
        
        return self


    def transform(self, X):
        """
        Makes predictions using the selected models and returns a dataframe containing those predictions.
        
        :param X: feature data based on which predictions are to be made
        :return: X_transformed - a dataframe with #columns = #selected models containing predictions
        :rtype: pandas dataframe
        """

        models =  ['GBR', 'LASSO', 'SVR', 'KNN']
        mask = [self.dogbr, self.dolasso, self.dosvr, self.doknn]        
        columns = list(it.compress(models, mask))
        index = np.arange(0, len(np.asarray(X)))
        X_transformed = pd.DataFrame(columns=columns, index=index)
        X_transformed = X_transformed.fillna(0)
        if self.dosvr | self.dolasso:
            X_scaled = self.scaler_.transform(X)        
        if self.dogbr:
            X_transformed['GBR'] = self.gbr_.predict(X)
        if self.dolasso: 
            X_transformed['LASSO'] = np.expm1(self.lasso_.predict(X_scaled))
        if self.dosvr:
            X_transformed['SVR'] = np.expm1(self.svr_.predict(X_scaled))
        if self.doknn:
            X_transformed['KNN'] = self.kn_.predict(X)

        return X_transformed 
            