import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import load_data
from train import train_model

class TestTraining:
    
    def test_dataset_loading(self):
        """Test if dataset loads correctly."""
        X_train, X_test, y_train, y_test, scaler = load_data()
        
        assert X_train.shape[0] > 0, "Training set should not be empty"
        assert X_test.shape[0] > 0, "Test set should not be empty"
        assert len(y_train) == X_train.shape[0], "Target and features should have same length"
        assert X_train.shape[1] == 8, "California housing should have 8 features"
    
    def test_model_creation(self):
        """Test if LinearRegression model is created correctly."""
        model = LinearRegression()
        assert isinstance(model, LinearRegression), "Model should be LinearRegression instance"
    
    def test_model_training(self):
        """Test if model training works and produces valid coefficients."""
        model, scaler, test_r2 = train_model()
        
        # Check if model is trained (has coefficients)
        assert hasattr(model, 'coef_'), "Trained model should have coefficients"
        assert hasattr(model, 'intercept_'), "Trained model should have intercept"
        assert len(model.coef_) == 8, "Should have 8 coefficients for 8 features"
    
    def test_model_performance(self):
        """Test if model meets minimum performance threshold."""
        model, scaler, test_r2 = train_model()
        
        # R² score should be reasonable for this dataset
        assert test_r2 > 0.5, f"R² score {test_r2:.4f} should be > 0.5"
        assert test_r2 < 1.0, f"R² score {test_r2:.4f} should be < 1.0"