import numpy as np
from utils import load_model, save_model
import joblib

def quantize_parameters():
    """Manually quantize model parameters to 8-bit unsigned integers."""
    print("Loading trained model...")
    model = load_model('trained_model.joblib')
    
    # Extract parameters
    coef = model.coef_
    intercept = model.intercept_
    
    print(f"Original coefficient range: [{coef.min():.6f}, {coef.max():.6f}]")
    print(f"Original intercept value: {intercept:.6f}")
    
    print(f"Original coefficients shape: {coef.shape}")
    print(f"Original intercept: {intercept}")
    
    # Save original parameters
    original_params = {
        'coef': coef,
        'intercept': intercept
    }
    joblib.dump(original_params, 'unquant_params.joblib')
    print("Original parameters saved to unquant_params.joblib")
    
    # Quantization process (improved based on reference)
    def quantize_weights_advanced(weights, bits=8):
        """Advanced quantization with better range handling."""
        # Normalize weights to [0, 1] range
        w_min = np.min(weights)
        w_max = np.max(weights)
        
        # Handle edge case where all weights are the same
        if abs(w_max - w_min) < 1e-8:
            range_val = max(abs(w_min) * 0.01, 1e-6)
            w_min = w_min - range_val
            w_max = w_max + range_val
        
        # Normalize to [0, 1]
        weights_norm = (weights - w_min) / (w_max - w_min)
        
        # Quantize to n-bit unsigned integers
        levels = 2**bits - 1
        quantized = np.round(weights_norm * levels).astype(np.uint8)
        
        return quantized, w_min, w_max
    
    def dequantize_weights_advanced(quantized, w_min, w_max, bits=8):
        """Dequantize weights back to original range."""
        levels = 2**bits - 1
        weights_norm = quantized.astype(np.float32) / levels
        return weights_norm * (w_max - w_min) + w_min
    
    # Quantize coefficients and intercept using improved method
    quant_coef, coef_min, coef_max = quantize_weights_advanced(coef)
    
    # For intercept, create a small variation to avoid quantization issues
    intercept_array = np.array([intercept])
    quant_intercept_arr, int_min, int_max = quantize_weights_advanced(intercept_array)
    quant_intercept = quant_intercept_arr[0]
    
    # Calculate storage savings
    original_size = coef.nbytes + np.array([intercept]).nbytes
    quantized_size = quant_coef.nbytes + np.array([quant_intercept]).nbytes
    compression_ratio = original_size / quantized_size
    
    print(f"Storage: Original={original_size} bytes, Quantized={quantized_size} bytes")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    # Save quantized parameters
    quantized_params = {
        'quant_coef': quant_coef,
        'coef_min': coef_min,
        'coef_max': coef_max,
        'quant_intercept': quant_intercept,
        'int_min': int_min,
        'int_max': int_max,
        'original_size': original_size,
        'quantized_size': quantized_size,
        'compression_ratio': compression_ratio
    }
    
    joblib.dump(quantized_params, 'quant_params.joblib')
    print("Quantized parameters saved to quant_params.joblib")
    
    # Test dequantization
    dequant_coef = dequantize_weights_advanced(quant_coef, coef_min, coef_max)
    dequant_intercept = dequantize_weights_advanced(
        np.array([quant_intercept]), int_min, int_max
    )[0]
    
    # Calculate quantization errors
    coef_error = np.mean(np.abs(coef - dequant_coef))
    intercept_error = abs(intercept - dequant_intercept)
    
    print(f"Quantization error in coefficients: {coef_error:.8f}")
    print(f"Quantization error in intercept: {intercept_error:.8f}")
    
    # Test inference with dequantized weights
    print("\nTesting inference with dequantized weights...")
    from utils import load_data, calculate_metrics
    X_train, X_test, y_train, y_test, scaler = load_data()
    
    # Original predictions
    original_pred = model.predict(X_test)
    
    # Dequantized predictions
    dequant_pred = X_test @ dequant_coef + dequant_intercept
    
    # Calculate performance metrics
    orig_r2, orig_mse = calculate_metrics(y_test, original_pred)
    quant_r2, quant_mse = calculate_metrics(y_test, dequant_pred)
    
    print(f"Original model  - R²: {orig_r2:.6f}, MSE: {orig_mse:.6f}")
    print(f"Quantized model - R²: {quant_r2:.6f}, MSE: {quant_mse:.6f}")
    print(f"Performance degradation - R²: {abs(orig_r2 - quant_r2):.6f}, MSE: {abs(orig_mse - quant_mse):.6f}")
    
    # Add performance metrics to saved parameters
    quantized_params.update({
        'original_r2': orig_r2,
        'original_mse': orig_mse,
        'quantized_r2': quant_r2,
        'quantized_mse': quant_mse,
        'r2_degradation': abs(orig_r2 - quant_r2),
        'mse_degradation': abs(orig_mse - quant_mse)
    })

if __name__ == "__main__":
    quantize_parameters()