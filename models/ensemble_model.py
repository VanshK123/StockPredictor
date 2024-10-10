import xgboost as xgb

def ensemble_with_xgboost(X_train, y_train, X_test):
    """
    Train XGBoost model with the combined feature set and predict for the test set.
    """
    model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
    model.fit(X_train, y_train)  # Train on the combined features

    # Make predictions on the test set
    predictions = model.predict(X_test)
    return predictions
