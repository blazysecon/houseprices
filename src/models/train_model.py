from sklearn.ensemble import GradientBoostingRegressor


def build_fit_model(X_train, y_train):
    gbr = GradientBoostingRegressor(max_depth=4, learning_rate=0.1, random_state=0)
    gbr.fit(X_train, y_train)
    return gbr

