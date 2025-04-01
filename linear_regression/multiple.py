"""
Multiple Linear Regression from Scratch using Matrix Algebra (OLS Normal Equation).

This script simulates data with two predictor variables and fits a multiple
linear regression model from scratch. It includes evaluation and visualization
(3D plane and residuals plot).
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for 3D plotting

# --------------------------------------
# Configuration & True Parameters
# --------------------------------------
# Define the 'true' model parameters we want the regression to estimate
TRUE_INTERCEPT = 5.0
TRUE_COEFFS = np.array([1.5, -2.0]) # Coefficients for x1 and x2 (beta_1, beta_2)
NUM_SAMPLES = 200
NOISE_STD_DEV = 2.0      # Standard deviation of the error term

# --------------------------------------
# Function Definitions
# --------------------------------------

def generate_synthetic_data_mlr(true_intercept, true_coeffs, num_samples, noise_std_dev):
    """
    Generates synthetic data for multiple linear regression.

    Args:
        true_intercept (float): The true intercept (beta_0).
        true_coeffs (np.array): Numpy array of true coefficients (beta_1, beta_2,...).
        num_samples (int): Number of data points.
        noise_std_dev (float): Standard deviation of the Gaussian noise.

    Returns:
        tuple: (X_raw, y)
               X_raw (np.array): Matrix of predictor variables (n_samples x n_predictors).
               y (np.array): Vector of the dependent variable (n_samples x 1).
    """
    print(f"Generating {num_samples} synthetic data samples...")
    num_predictors = len(true_coeffs)

    # Generate predictor variables (e.g., uniformly distributed)
    X_raw = np.random.rand(num_samples, num_predictors) * 10 # Predictors between 0 and 10

    # Calculate the dependent variable using the linear model + noise
    epsilon = np.random.normal(0, noise_std_dev, num_samples)
    # Linear combination: intercept + X_raw @ true_coeffs
    y = true_intercept + X_raw @ true_coeffs + epsilon
    # Reshape y to be a column vector (n x 1)
    y = y.reshape(-1, 1)

    print(f"Shape of predictor matrix (X_raw): {X_raw.shape}")
    print(f"Shape of dependent variable vector (y): {y.shape}\n")
    return X_raw, y

def multiple_linear_regression_scratch(X_raw, y):
    """
    Calculates multiple linear regression coefficients using the OLS Normal Equation.

    Args:
        X_raw (np.array): Matrix of predictor variables (n_samples x n_predictors).
                          Does NOT include the intercept column yet.
        y (np.array): Vector of the dependent variable (n_samples x 1).

    Returns:
        np.array: Vector of estimated coefficients (beta_0, beta_1, ... beta_p),
                  size ((p+1) x 1). Returns None if calculation fails.
    """
    print("Calculating MLR coefficients using Normal Equation...")
    n_samples = X_raw.shape[0]
    if n_samples == 0 or n_samples != y.shape[0]:
        raise ValueError("Invalid input shapes or empty arrays.")

    # 1. Construct the Design Matrix X by adding a column of ones for the intercept
    X_design = np.insert(X_raw, 0, 1, axis=1) # Insert 1s at column index 0
    # Alternative: X_design = np.c_[np.ones(n_samples), X_raw]
    print(f"Shape of Design Matrix (X_design): {X_design.shape}")

    # 2. Calculate coefficients using the Normal Equation: beta = (X^T * X)^(-1) * X^T * y
    try:
        XT = X_design.T
        XTX = XT @ X_design
        XTX_inv = np.linalg.inv(XTX) # Calculate the inverse of (X^T * X)
        XTY = XT @ y
        beta_hat = XTX_inv @ XTY     # Calculate the coefficient vector

        print("Coefficient calculation successful.")
        print(f"Shape of estimated coefficient vector (beta_hat): {beta_hat.shape}\n")
        return beta_hat

    except np.linalg.LinAlgError:
        # This happens if XTX is singular (not invertible) -> perfect multicollinearity
        print("Error: Matrix (X^T * X) is singular. Cannot compute inverse.")
        print("This often indicates perfect multicollinearity among predictors.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during coefficient calculation: {e}")
        return None

def predict_mlr(X_raw_new, coefficients):
    """
    Makes predictions using the trained multiple linear regression model.

    Args:
        X_raw_new (np.array): New predictor variable values (n_new_samples x n_predictors).
                              Does NOT include the intercept column.
        coefficients (np.array): Estimated coefficient vector (beta_0, beta_1,...).

    Returns:
        np.array: Predicted dependent variable values (n_new_samples x 1).
    """
    if coefficients is None:
        raise ValueError("Coefficients are None, cannot make predictions.")
    
    # Add column of ones for the intercept to the new data
    X_design_new = np.insert(X_raw_new, 0, 1, axis=1)
    
    # Predict: y_hat = X_design_new @ coefficients
    y_predicted = X_design_new @ coefficients
    return y_predicted

def evaluate_mlr_model(y_true, y_predicted, num_predictors, num_samples):
    """
    Calculates MSE, RMSE, R-squared, and Adjusted R-squared for MLR.

    Args:
        y_true (np.array): Actual true values (n_samples x 1).
        y_predicted (np.array): Predicted values (n_samples x 1).
        num_predictors (int): Number of predictor variables (p, excluding intercept).
        num_samples (int): Number of samples (n).

    Returns:
        dict: Dictionary containing 'mse', 'rmse', 'r2', 'adj_r2'.
    """
    if num_samples <= num_predictors + 1:
         print("Warning: Not enough samples to reliably calculate Adjusted R². Returning NaN.")
         adj_r2 = np.nan # Degrees of freedom issue
    
    # Ensure inputs are flat arrays for some calculations if needed, or handle (n,1) shape
    y_true_flat = y_true.flatten()
    y_predicted_flat = y_predicted.flatten()

    # MSE
    mse = np.mean((y_true_flat - y_predicted_flat)**2)
    # RMSE
    rmse = np.sqrt(mse)

    # R-squared
    sst = np.sum((y_true_flat - np.mean(y_true_flat))**2)
    sse = np.sum((y_true_flat - y_predicted_flat)**2)
    r2 = 1 - (sse / sst) if sst != 0 else 0

    # Adjusted R-squared
    if num_samples > num_predictors + 1 and sst != 0:
         adj_r2 = 1 - ( (1 - r2) * (num_samples - 1) / (num_samples - num_predictors - 1) )
    else:
         # Handle cases where calculation is not possible or meaningful
         adj_r2 = np.nan if sst !=0 else 0 


    return {'mse': mse, 'rmse': rmse, 'r2': r2, 'adj_r2': adj_r2}

def plot_mlr_results(X_raw, y_true, y_predicted, coefficients):
    """
    Generates plots for MLR results (assuming 2 predictors for 3D plot).
    Includes:
    1. 3D scatter plot of data and the fitted regression plane.
    2. Residuals vs. Fitted values plot.
    """
    num_predictors = X_raw.shape[1]
    residuals = y_true.flatten() - y_predicted.flatten()

    # --- Plot 1: Residuals vs. Fitted ---
    plt.figure(figsize=(10, 5))
    plt.scatter(y_predicted.flatten(), residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title('Residuals vs. Fitted Values')
    plt.xlabel('Fitted Values (ŷ)')
    plt.ylabel('Residuals (y - ŷ)')
    plt.grid(True)
    plt.savefig("./assests/plots/mlr_residuals_plot.png")
    print("\nResiduals vs. Fitted plot saved to ./assests/plots/mlr_residuals_plot.png")
    # plt.show() # Uncomment to display interactively

    # --- Plot 2: 3D Scatter Plot and Regression Plane (only if p=2) ---
    if num_predictors == 2:
        print("Generating 3D plot (requires 2 predictors)...")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot of actual data
        x1 = X_raw[:, 0]
        x2 = X_raw[:, 1]
        ax.scatter(x1, x2, y_true.flatten(), c='blue', marker='o', alpha=0.6, label='Actual Data')

        # Create a grid for the plane
        x1_surf = np.linspace(x1.min(), x1.max(), 10)
        x2_surf = np.linspace(x2.min(), x2.max(), 10)
        x1_surf, x2_surf = np.meshgrid(x1_surf, x2_surf)

        # Predict y values for the grid points to define the plane
        X_surf_raw = np.c_[x1_surf.ravel(), x2_surf.ravel()] # Combine grid points
        fitted_surf = predict_mlr(X_surf_raw, coefficients)
        fitted_surf = fitted_surf.reshape(x1_surf.shape) # Reshape back to grid shape

        # Plot the regression plane
        ax.plot_surface(x1_surf, x2_surf, fitted_surf, color='red', alpha=0.4, label='Fitted Plane')
        
        # Setting labels slightly away from the axes planes using labelpad
        ax.set_xlabel('Predictor x1', labelpad=10)
        ax.set_ylabel('Predictor x2', labelpad=10)
        ax.set_zlabel('Dependent y', labelpad=10)
        ax.set_title('Multiple Linear Regression Fit (2 Predictors)')
        
        # Dummy proxy artists for legend (since plot_surface doesn't directly support labels well)
        # proxy_actual = plt.Rectangle((0, 0), 1, 1, fc="blue", alpha=0.6)
        # proxy_fitted = plt.Rectangle((0, 0), 1, 1, fc="red", alpha=0.4)
        # ax.legend([proxy_actual, proxy_fitted], ['Actual Data', 'Fitted Plane']) # Can sometimes be buggy in 3D

        plt.savefig("./assests/plots/mlr_3d_plane_plot.png")
        print("3D plot saved to ./assests/plots/mlr_3d_plane_plot.png")
        # plt.show() # Uncomment to display interactively
    else:
        print("\nSkipping 3D plane plot (requires exactly 2 predictors).")


# --------------------------------------
# Main Execution Block
# --------------------------------------
if __name__ == "__main__":

    # 1. Generate Data
    X_data_raw, y_data = generate_synthetic_data_mlr(
        TRUE_INTERCEPT, TRUE_COEFFS, NUM_SAMPLES, NOISE_STD_DEV
    )
    num_predictors_p = X_data_raw.shape[1] # Number of predictors 'p'

    # 2. Train Model (Calculate Coefficients)
    estimated_coefficients = multiple_linear_regression_scratch(X_data_raw, y_data)

    if estimated_coefficients is not None:
        print("Estimated Coefficients (beta_hat):")
        print(f"  Intercept (beta_0): {estimated_coefficients[0][0]:.4f} (True: {TRUE_INTERCEPT})")
        for i in range(num_predictors_p):
            print(f"  Coef for x{i+1} (beta_{i+1}): {estimated_coefficients[i+1][0]:.4f} (True: {TRUE_COEFFS[i]})")

        # 3. Make Predictions (on training data to evaluate fit)
        y_predicted = predict_mlr(X_data_raw, estimated_coefficients)

        # 4. Evaluate Model
        metrics = evaluate_mlr_model(y_data, y_predicted, num_predictors_p, NUM_SAMPLES)

        print("\nModel Evaluation:")
        print(f"  Mean Squared Error (MSE):      {metrics['mse']:.4f}")
        print(f"  Root Mean Squared Error (RMSE):{metrics['rmse']:.4f}")
        print(f"  R-squared (R²):                {metrics['r2']:.4f}")
        print(f"  Adjusted R-squared (Adj. R²):  {metrics['adj_r2']:.4f}")

        # 5. Visualize Results
        plot_mlr_results(X_data_raw, y_data, y_predicted, estimated_coefficients)

    else:
        print("\nModel training failed. Cannot proceed with evaluation or plotting.")