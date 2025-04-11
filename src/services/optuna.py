import optuna
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

def objective(trial, preprocessor, X_train, y_train):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
    }

    model = XGBRegressor(objective='reg:squarederror', **params)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    score = cross_val_score(pipeline, X_train, y_train, scoring='neg_root_mean_squared_error', cv=3).mean()
    return -score