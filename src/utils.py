import numpy as np
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(normalization='standard'):
    """Load and prepare California Housing dataset with different normalization options."""
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    print(f"Original data range: [{X.min():.3f}, {X.max():.3f}]")
    
    # Split the data first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Apply different normalization strategies
    if normalization == 'standard':
        # Standard scaling (zero mean, unit variance)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    elif normalization == 'minmax':
        # Min-max scaling to [0, 1]
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    elif normalization == 'robust':
        # Robust scaling (less sensitive to outliers)
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        # No scaling
        scaler = None
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    print(f"Processed data range: [{X_train_scaled.min():.3f}, {X_train_scaled.max():.3f}]")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_model(model, filename):
    """Save model using joblib."""
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename):
    """Load model using joblib."""
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

def calculate_metrics(y_true, y_pred):
    """Calculate R2 score and MSE."""
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return r2, mse