"""
Polynomial Regression from Scratch for Predictive Maintenance (RUL Prediction).

This script simulates non-linear RUL data based on operating hours
and fits a polynomial regression model from scratch by transforming features
and using the OLS Normal Equation for Multiple Linear Regression.
"""

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------
# Configuration
# --------------------------------------
POLY_DEGREE = 2          # Degree of the polynomial to fit (e.g., 2 for quadratic)
NUM_SAMPLES = 150
NOISE_STD_DEV = 40.0     # Noise level in RUL data

# True underlying relationship (quadratic for demonstration)
# RUL = TRUE_C + TRUE_B1*hours + TRUE_B2*hours^2 + noise
TRUE_C = 1200.0
TRUE_B1 = -0.2
TRUE_B2 = -0.0002 # Negative coefficient for hours^2 makes RUL decrease faster over time

# PdM Example Config
CURRENT_OPERATING_HOURS = 1100
MAINTENANCE_THRESHOLD = 150

# --------------------------------------
# Function Definitions
# --------------------------------------

def generate_non_linear_rul_data(c, b1, b2, num_samples, noise_std_dev):
    """Generates synthetic RUL data with a quadratic relationship + noise."""
    print(f"Generating {num_samples} synthetic non-linear RUL samples...")
    # Generate operating hours (independent variable x)
    X_hours = np.random.uniform(0, 1500, num_samples) # Operating hours from 0 to 1500

    # Calculate RUL using the true quadratic relationship + noise
    epsilon = np.random.normal(0, noise_std_dev, num_samples)
    y_rul = c + b1 * X_hours + b2 * (X_hours**2) + epsilon

    # Ensure RUL is not negative
    y_rul = np.maximum(y_rul, 0)

    print(f"Shape of operating hours (X_hours): {X_hours.shape}")
    print(f"Shape of RUL (y_rul): {y_rul.shape}\n")
    # Return X as a column vector for consistency later, y as a column vector too
    return X_hours.reshape(-1, 1), y_rul.reshape(-1, 1)

def create_polynomial_features(X, degree):
    """
    Generates polynomial features for the input matrix X.

    Args:
        X (np.array): Input data (n_samples x 1). Assumes single predictor.
        degree (int): The maximum degree of the polynomial features.

    Returns:
        np.array: Matrix with polynomial features [x, x^2, ..., x^degree]
                  Shape: (n_samples x degree).
    """
    if X.shape[1] != 1:
        raise ValueError("Input X should have only one column (single predictor)")
    
    X_poly = X # Start with the original feature x^1
    if degree > 1:
        for d in range(2, degree + 1):
            X_poly = np.hstack((X_poly, np.power(X, d)))
            # Alternative: np.c_[X_poly, X**d]
    return X_poly

def fit_polynomial_regression(X_orig, y, degree):
    """
    Fits a polynomial regression model using the OLS Normal Equation.

    Args:
        X_orig (np.array): Original independent variable data (n_samples x 1).
        y (np.array): Dependent variable data (n_samples x 1).
        degree (int): Degree of the polynomial to fit.

    Returns:
        np.array: Estimated coefficient vector (beta_0, beta_1, ... beta_d),
                  shape ((degree+1) x 1). Returns None if calculation fails.
    """
    print(f"Fitting Polynomial Regression of degree {degree}...")

    # 1. Create polynomial features from the original X
    X_poly_features = create_polynomial_features(X_orig, degree)
    print(f"Shape of polynomial features matrix: {X_poly_features.shape}")

    # 2. Construct the full Design Matrix X by adding intercept column
    n_samples = X_orig.shape[0]
    X_design = np.c_[np.ones((n_samples, 1)), X_poly_features]
    print(f"Shape of Design Matrix (X_design): {X_design.shape}")

    # 3. Solve using the Normal Equation (same as MLR)
    try:
        XT = X_design.T
        XTX = XT @ X_design
        XTX_inv = np.linalg.inv(XTX)
        XTY = XT @ y
        beta_hat = XTX_inv @ XTY

        print("Coefficient calculation successful.")
        print(f"Shape of estimated coefficient vector (beta_hat): {beta_hat.shape}\n")
        return beta_hat

    except np.linalg.LinAlgError:
        print("Error: Matrix (X^T * X) is singular. Cannot compute inverse.")
        print("Consider checking for multicollinearity or issues with high degrees/scaling.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during coefficient calculation: {e}")
        return None

def predict_poly(X_new_orig, coefficients, degree):
    """
    Makes predictions using the trained polynomial regression model.

    Args:
        X_new_orig (np.array): New original independent variable values (n_new x 1).
        coefficients (np.array): Estimated coefficient vector (beta_0, ... beta_d).
        degree (int): The degree of the polynomial model that was fitted.

    Returns:
        np.array: Predicted dependent variable values (n_new x 1).
    """
    if coefficients is None:
        raise ValueError("Coefficients are None, cannot make predictions.")
    
    # 1. Create polynomial features for the new data
    X_new_poly_features = create_polynomial_features(X_new_orig, degree)
    
    # 2. Add intercept column to create the new design matrix
    X_new_design = np.c_[np.ones((X_new_orig.shape[0], 1)), X_new_poly_features]
    
    # 3. Predict: y_hat = X_new_design @ coefficients
    y_predicted = X_new_design @ coefficients
    return y_predicted

# Evaluation function (can reuse the one from MLR, p = degree)
def evaluate_poly_model(y_true, y_predicted, degree, num_samples):
    """Calculates MSE, RMSE, R-squared, and Adjusted R-squared for Polynomial Regression."""
    num_predictors = degree # In polynomial regression, p = degree
    # Use the same logic as evaluate_mlr_model
    # (Ensure y_true and y_predicted are flattened or handled correctly)
    y_true_flat = y_true.flatten()
    y_predicted_flat = y_predicted.flatten()

    mse = np.mean((y_true_flat - y_predicted_flat)**2)
    rmse = np.sqrt(mse)
    sst = np.sum((y_true_flat - np.mean(y_true_flat))**2)
    sse = np.sum((y_true_flat - y_predicted_flat)**2)
    r2 = 1 - (sse / sst) if sst != 0 else 0

    if num_samples > num_predictors + 1 and sst != 0:
         adj_r2 = 1 - ( (1 - r2) * (num_samples - 1) / (num_samples - num_predictors - 1) )
    else:
         adj_r2 = np.nan if sst !=0 else 0

    return {'mse': mse, 'rmse': rmse, 'r2': r2, 'adj_r2': adj_r2}

def plot_poly_regression(X_orig, y_true, coefficients, degree, filename="./assests/plots/poly_regression_pdm.png"):
    """Plots the original data and the fitted polynomial curve."""
    
    plt.figure(figsize=(10, 6))
    # Scatter plot of original data
    plt.scatter(X_orig.flatten(), y_true.flatten(), alpha=0.6, label='Actual RUL Data')

    # Generate points for the fitted curve
    X_line = np.linspace(X_orig.min(), X_orig.max(), 200).reshape(-1, 1)
    y_line = predict_poly(X_line, coefficients, degree)
    plt.plot(X_line.flatten(), y_line.flatten(), color='red', linewidth=2, label=f'Fitted Polynomial (degree {degree})')

    plt.title('Polynomial Regression: RUL vs Operating Hours')
    plt.xlabel('Operating Hours (X)')
    plt.ylabel('Remaining Useful Life (RUL) (y)')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0) # RUL shouldn't be negative

    try:
        plt.savefig(filename)
        print(f"\nPlot saved to {filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    # plt.show()

# --- Residual Plot Function (reuse or adapt from MLR script if needed) ---
def plot_residuals(y_true, y_predicted, filename="./assests/plots/poly_residuals_plot.png"):
    """Generates and saves a residuals vs. fitted values plot."""
    residuals = y_true.flatten() - y_predicted.flatten()
    fitted_values = y_predicted.flatten()

    plt.figure(figsize=(10, 5))
    plt.scatter(fitted_values, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title('Residuals vs. Fitted Values (Polynomial Model)')
    plt.xlabel('Fitted Values (ŷ)')
    plt.ylabel('Residuals (y - ŷ)')
    plt.grid(True)
    try:
        plt.savefig(filename)
        print(f"Residual plot saved to {filename}")
    except Exception as e:
        print(f"Error saving residual plot: {e}")
    # plt.show()

# --------------------------------------
# Main Execution Block
# --------------------------------------
if __name__ == "__main__":

    # 1. Generate Non-Linear Data
    X_hours_data, y_rul_data = generate_non_linear_rul_data(
        TRUE_C, TRUE_B1, TRUE_B2, NUM_SAMPLES, NOISE_STD_DEV
    )

    # 2. Fit Polynomial Regression Model
    estimated_coefficients = fit_polynomial_regression(X_hours_data, y_rul_data, POLY_DEGREE)

    if estimated_coefficients is not None:
        print("Estimated Coefficients (beta_hat):")
        print(f"  Intercept (beta_0): {estimated_coefficients[0][0]:.4f}")
        for i in range(POLY_DEGREE):
            print(f"  Coef for x^{i+1} (beta_{i+1}): {estimated_coefficients[i+1][0]:.4f}")
        # Note: Comparing estimated coefficients directly to true ones for higher orders
        # can be tricky if features weren't scaled, but we can see if they have the right signs/magnitudes.

        # 3. Make Predictions (on training data for evaluation)
        y_predicted = predict_poly(X_hours_data, estimated_coefficients, POLY_DEGREE)

        # 4. Evaluate Model
        metrics = evaluate_poly_model(y_rul_data, y_predicted, POLY_DEGREE, NUM_SAMPLES)

        print("\nModel Evaluation:")
        print(f"  Polynomial Degree:             {POLY_DEGREE}")
        print(f"  Mean Squared Error (MSE):      {metrics['mse']:.2f}")
        print(f"  Root Mean Squared Error (RMSE):{metrics['rmse']:.2f}")
        print(f"  R-squared (R²):                {metrics['r2']:.4f}")
        print(f"  Adjusted R-squared (Adj. R²):  {metrics['adj_r2']:.4f}")

        # 5. Visualize Results
        plot_poly_regression(X_hours_data, y_rul_data, estimated_coefficients, POLY_DEGREE)
        plot_residuals(y_rul_data, y_predicted) # Check residuals for the polynomial fit

        # 6. Predictive Maintenance Example
        print(f"\nPredictive Example:")
        current_hours_val = np.array([[CURRENT_OPERATING_HOURS]]) # Needs to be 2D array (1x1)
        predicted_rul = predict_poly(current_hours_val, estimated_coefficients, POLY_DEGREE)
        predicted_rul_scalar = max(0, predicted_rul[0][0]) # Extract scalar, ensure non-negative

        print(f"  For a component with {CURRENT_OPERATING_HOURS} operating hours...")
        print(f"  The estimated RUL using polynomial regression (degree {POLY_DEGREE}) is: {predicted_rul_scalar:.2f} hours")

        # Maintenance Threshold Check
        if predicted_rul_scalar < MAINTENANCE_THRESHOLD:
            print(f"  ALERT! Estimated RUL ({predicted_rul_scalar:.2f}) is below the threshold ({MAINTENANCE_THRESHOLD} hours).")
            print("  Recommend scheduling maintenance.")
        else:
            print(f"  Estimated RUL ({predicted_rul_scalar:.2f}) is above the threshold ({MAINTENANCE_THRESHOLD} hours).")
            print("  Continue monitoring.")

    else:
        print("\nModel training failed. Cannot proceed.")