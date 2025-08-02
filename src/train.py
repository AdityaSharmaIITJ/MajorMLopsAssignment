from sklearn.linear_model import LinearRegression
from utils import load_data, save_model, calculate_metrics
import numpy as np

def train_model():
    """Train Linear Regression model on California Housing dataset."""
    print("Loading California Housing dataset...")
    X_train, X_test, y_train, y_test, scaler = load_data()
    
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_r2, train_mse = calculate_metrics(y_train, y_train_pred)
    test_r2, test_mse = calculate_metrics(y_test, y_test_pred)
    
    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Test R² Score: {test_r2:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    
    # Save model and scaler
    save_model(model, 'trained_model.joblib')
    save_model(scaler, 'scaler.joblib')
    
    return model, scaler, test_r2

if __name__ == "__main__":
    train_model()