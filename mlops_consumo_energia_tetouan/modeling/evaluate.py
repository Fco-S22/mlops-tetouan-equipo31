from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def evaluate_regression(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
    }
