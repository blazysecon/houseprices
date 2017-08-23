from StackingTransformer import StackingTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV

def build_fit_pipe(X, y):        
    """
    Builds and fits a 3-stage pipeline on the training data passed as argument
    Returns the fitted pipeline.
    
    The 3 stages of the pipe are:
        - Feature selection using Recursive Feature Elimination with Cross Validation
        - Transformation - this stage actually takes the features and produces predictions with up to 4 different models
        - Regression - a linear regression model is trained on the output of the different models and produces a final prediction

    The Transformation stage is executed using the custom Transformer StackingTransformer.
    
    :param X: feature data to which to fit the pipe
    :param y: target data to which to fit the pipe
    :return: fitted pipeline
    :rtype: Pipeline object fitted to X,y

    
    """   
    rfecvgbr = GradientBoostingRegressor()
    rfecv = RFECV(estimator=rfecvgbr, step=1)    
    st = StackingTransformer(True, True, False, False)    
    lr = LinearRegression()   
    pipe = Pipeline([("reducer", rfecv), ("stackTransformer", st), ("linearRegression", lr)])    
    pipe.fit(X, y)
    return pipe

