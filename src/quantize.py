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
    
    print(f"Original coefficients shape: {coef.shape}")
    print(f"Original intercept: {intercept}")
    
    # Save original parameters
    original_params = {
        'coef': coef,
        'intercept': intercept
    }
    joblib.dump(original_params, 'unquant_params.joblib')
    print("Original parameters saved to unquant_params.joblib")
    
    # Quantization process
    def quantize_array(arr, bits=8):
        """Quantize array to specified bits."""
        # Find min and max values
        min_val = np.min(arr)
        max_val = np.max(arr)
        
        # Calculate scale and zero point
        scale = (max_val - min_val) / (2**bits - 1)
        zero_point = -min_val / scale
        
        # Quantize
        quantized = np.round((arr - min_val) / scale).astype(np.uint8)
        
        return quantized, scale, zero_point, min_val, max_val
    
    # Quantize coefficients
    quant_coef, coef_scale, coef_zero_point, coef_min, coef_max = quantize_array(coef)
    
    # Quantize intercept (treat as single value array)
    intercept_arr = np.array([intercept])
    quant_intercept, int_scale, int_zero_point, int_min, int_max = quantize_array(intercept_arr)
    
    # Save quantized parameters
    quantized_params = {
        'quant_coef': quant_coef,
        'coef_scale': coef_scale,
        'coef_zero_point': coef_zero_point,
        'coef_min': coef_min,
        'coef_max': coef_max,
        'quant_intercept': quant_intercept[0],
        'int_scale': int_scale,
        'int_zero_point': int_zero_point,
        'int_min': int_min,
        'int_max': int_max
    }
    
    joblib.dump(quantized_params, 'quant_params.joblib')
    print("Quantized parameters saved to quant_params.joblib")
    
    # Test dequantization
    def dequantize_array(quantized, scale, zero_point, min_val):
        """Dequantize array."""
        return quantized.astype(np.float32) * scale + min_val
    
    # Dequantize
    dequant_coef = dequantize_array(quant_coef, coef_scale, coef_zero_point, coef_min)
    dequant_intercept = dequantize_array(
        np.array([quant_intercept]), int_scale, int_zero_point, int_min
    )[0]
    
    print(f"Quantization error in coefficients: {np.mean(np.abs(coef - dequant_coef))}")
    print(f"Quantization error in intercept: {abs(intercept - dequant_intercept)}")
    
    # Test inference with dequantized weights
    print("\nTesting inference with dequantized weights...")
    from utils import load_data
    X_train, X_test, y_train, y_test, scaler = load_data()
    
    # Original predictions
    original_pred = model.predict(X_test[:5])
    
    # Dequantized predictions
    dequant_pred = X_test[:5] @ dequant_coef + dequant_intercept
    
    print("Original predictions (first 5):")
    print(original_pred)
    print("Dequantized predictions (first 5):")
    print(dequant_pred)
    print(f"Prediction difference: {np.mean(np.abs(original_pred - dequant_pred))}")

if __name__ == "__main__":
    quantize_parameters()