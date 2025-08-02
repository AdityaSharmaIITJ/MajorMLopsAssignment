from utils import load_model, load_data
import numpy as np

def predict_samples():
    """Load model and make predictions on test samples."""
    print("Loading trained model and data...")
    model = load_model('src/trained_model.joblib')
    scaler = load_model('src/scaler.joblib')
    
    # Load test data
    X_train, X_test, y_train, y_test, _ = load_data()
    
    # Make predictions on first 10 test samples
    predictions = model.predict(X_test[:10])
    actual = y_test[:10]
    
    print("\nSample Predictions vs Actual Values:")
    print("-" * 40)
    for i in range(10):
        print(f"Sample {i+1}: Predicted={predictions[i]:.3f}, Actual={actual[i]:.3f}")
    
    # Calculate accuracy metrics
    from utils import calculate_metrics
    all_predictions = model.predict(X_test)
    r2, mse = calculate_metrics(y_test, all_predictions)
    
    print(f"\nOverall Test Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    
    print("Model verification completed successfully!")

if __name__ == "__main__":
    predict_samples()