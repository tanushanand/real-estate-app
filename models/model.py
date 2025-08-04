"""
Model training and prediction functions for the Real Estate App.
"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_model(X, y, stratify_column=None):
    """
    Train a Linear Regression model and return the model and metrics.

    Parameters:
        X (pd.DataFrame): Feature set
        y (pd.Series): Target variable (price)
        stratify_column (pd.Series): Optional stratification column

    Returns:
        model (LinearRegression): Trained model
        metrics (dict): Performance metrics on test data
        X_test (pd.DataFrame): Test features for future prediction
        y_test (pd.Series): True target values
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=stratify_column if stratify_column is not None else None
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': mean_squared_error(y_test, y_pred, squared=False),
            'R2': r2_score(y_test, y_pred)
        }

        return model, metrics, X_test, y_test

    except Exception as e:
        raise Exception(f"Error during model training: {e}")

def predict(model, input_df):
    """
    Predict house prices using the trained model.

    Parameters:
        model (LinearRegression): Trained model
        input_df (pd.DataFrame): DataFrame of input features

    Returns:
        np.ndarray: Predicted price(s)
    """
    try:
        return model.predict(input_df)
    except Exception as e:
        raise Exception(f"Error during prediction: {e}")
