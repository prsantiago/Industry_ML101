"""
Simple Linear Regression from Scratch for Predictive Maintenance Simulation.

This script simulates Remaining Useful Life (RUL) data based on operating hours
and fits a simple linear regression model from scratch using OLS formulas.
It then evaluates the model and shows a predictive example.
"""

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------
# Configuration & True Parameters
# --------------------------------------
# These are the 'real' parameters of the system we are simulating
TRUE_INTERCEPT = 1000  # Theoretical initial RUL (in hours) when hours = 0
TRUE_SLOPE = -0.5      # RUL hours lost per operating hour
NUM_SAMPLES = 100      # Number of data points to generate
NOISE_STD_DEV = 50     # Standard deviation of the noise (uncertainty)

# Predictive Maintenance Example Config
CURRENT_OPERATING_HOURS = 800
MAINTENANCE_THRESHOLD = 200 # RUL threshold (in hours) to trigger maintenance

# --------------------------------------
# Function Definitions
# --------------------------------------

def generate_synthetic_data(true_intercept, true_slope, num_samples, noise_std_dev):
    """
    Generates synthetic data for a predictive maintenance scenario.

    Args:
        true_intercept (float): The true intercept (beta_0) of the linear relationship.
        true_slope (float): The true slope (beta_1) of the linear relationship.
        num_samples (int): The number of data points to generate.
        noise_std_dev (float): The standard deviation of the Gaussian noise to add.

    Returns:
        tuple: (X, y) - Numpy arrays for independent variable (operating hours)
               and dependent variable (RUL).
    """
    print(f"Generating {num_samples} synthetic data samples...")
    # Generate operating hours (independent variable X)
    # Ensure it doesn't start at 0 to make the intercept non-trivial visually
    X = np.random.uniform(50, 1500, num_samples) # Hours between 50 and 1500

    # Generate RUL (dependent variable y) with Gaussian noise
    # y = beta_0 + beta_1 * x + epsilon
    epsilon = np.random.normal(0, noise_std_dev, num_samples)
    y = true_intercept + true_slope * X + epsilon

    # Ensure RUL is not negative (physical constraint)
    y = np.maximum(y, 0)

    print(f"First 5 operating hours (X): {X[:5]}")
    print(f"First 5 RUL values (y): {y[:5]}\n")

    return X, y

def simple_linear_regression_scratch(x_data, y_data):
    """
    Calculates simple linear regression coefficients (intercept, slope)
    using the Ordinary Least Squares (OLS) analytical formulas.

    Args:
        x_data (np.array): Numpy array of the independent variable.
        y_data (np.array): Numpy array of the dependent variable.

    Returns:
        tuple: (intercept, slope) - Estimated intercept (beta_0) and slope (beta_1).
    """
    print("Calculating regression coefficients from scratch...")

    n = len(x_data)
    if n == 0:
        raise ValueError("Input arrays cannot be empty.")
    if n != len(y_data):
        raise ValueError("Input arrays must have the same length.")

    # Calculate means
    x_mean = np.mean(x_data)
    y_mean = np.mean(y_data)
    print(f"  Mean of X (hours): {x_mean:.2f}")
    print(f"  Mean of y (RUL): {y_mean:.2f}")

    # Calculate terms for the slope (beta_1)
    # Numerator: Sum of (x_i - x_mean) * (y_i - y_mean) -> SPXY (Sum of Products)
    # Denominator: Sum of (x_i - x_mean)^2 -> SSX (Sum of Squares X)
    numerator = np.sum((x_data - x_mean) * (y_data - y_mean))
    denominator = np.sum((x_data - x_mean)**2)
    print(f"  Numerator (SPXY): {numerator:.2f}")
    print(f"  Denominator (SSX): {denominator:.2f}")

    if denominator == 0:
        # This would happen if all X values are the same.
        raise ValueError("Variance of X is zero. Cannot compute slope.")

    # Calculate slope (beta_1)
    slope = numerator / denominator
    print(f"  Estimated Slope (beta_1): {slope:.4f}")

    # Calculate intercept (beta_0)
    # beta_0 = y_mean - beta_1 * x_mean
    intercept = y_mean - slope * x_mean
    print(f"  Estimated Intercept (beta_0): {intercept:.4f}\n")

    return intercept, slope

def predict(X_new, intercept, slope):
    """
    Makes predictions using the trained simple linear regression model.
    ŷ = intercept + slope * x

    Args:
        X_new (np.array or float): New independent variable value(s).
        intercept (float): Estimated intercept (beta_0).
        slope (float): Estimated slope (beta_1).

    Returns:
        np.array or float: Predicted dependent variable value(s).
    """
    return intercept + slope * X_new

def evaluate_model(y_true, y_predicted):
    """
    Calculates basic evaluation metrics: Mean Squared Error (MSE) and R-squared (R²).

    Args:
        y_true (np.array): Actual true values of the dependent variable.
        y_predicted (np.array): Values predicted by the model.

    Returns:
        tuple: (mse, r2_score)
    """
    # Mean Squared Error (MSE)
    # Average of the squared differences between actual and predicted values.
    mse = np.mean((y_true - y_predicted)**2)

    # R-squared (Coefficient of Determination)
    # Proportion of the variance in 'y' that is explained by 'x'.
    # R² = 1 - (Sum of Squared Errors / Total Sum of Squares)
    # Total Sum of Squares (SST): Total variance of y around its mean.
    sst = np.sum((y_true - np.mean(y_true))**2)
    # Sum of Squared Errors (SSE) - implicitly calculated for MSE
    sse = np.sum((y_true - y_predicted)**2)

    if sst == 0:
        # Handle the case where y_true is constant
        print("Warning: Total Sum of Squares (SST) is zero. R² is undefined, returning 0.")
        r2_score = 0
    else:
         r2_score = 1 - (sse / sst)

    return mse, r2_score

def plot_regression(X_data, y_data, intercept, slope, filename="./linear_regression/linear_regression_pdm.png"):
    """
    Generates and saves a plot of the original data and the regression line.

    Args:
        X_data (np.array): Independent variable data.
        y_data (np.array): Dependent variable data.
        intercept (float): Estimated intercept of the regression line.
        slope (float): Estimated slope of the regression line.
        filename (str): Name of the file to save the plot.
    """
    plt.figure(figsize=(10, 6))
    # Original data points (scatter plot)
    plt.scatter(X_data, y_data, alpha=0.7, label='Actual Data (Hours vs RUL)')

    # Regression line (predictions)
    # Generate points for the line across the range of X
    X_line = np.linspace(X_data.min(), X_data.max(), 100)
    y_line = predict(X_line, intercept, slope)
    plt.plot(X_line, y_line, color='red', linewidth=2,
             label=f'Linear Regression (ŷ = {intercept:.2f} {slope:+.2f}x)') # Added '+' for slope sign

    plt.title('Simple Linear Regression: RUL vs Operating Hours')
    plt.xlabel('Operating Hours (X)')
    plt.ylabel('Remaining Useful Life (RUL) (y)')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to a file
    try:
        plt.savefig(filename)
        print(f"\nPlot saved to {filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")


# --------------------------------------
# Main Execution Block
# --------------------------------------

if __name__ == "__main__":

    # 1. Generate Data
    X, y = generate_synthetic_data(TRUE_INTERCEPT, TRUE_SLOPE, NUM_SAMPLES, NOISE_STD_DEV)

    # 2. Train Model (Calculate Coefficients)
    try:
        est_intercept, est_slope = simple_linear_regression_scratch(X, y)

        print(f"Estimated Coefficients:")
        print(f"  Intercept (beta_0): {est_intercept:.4f} (True: {TRUE_INTERCEPT})")
        print(f"  Slope     (beta_1): {est_slope:.4f} (True: {TRUE_SLOPE})")

        # 3. Make Predictions (on training data to evaluate fit)
        y_predicted = predict(X, est_intercept, est_slope)

        # 4. Evaluate Model
        mse_value, r2_value = evaluate_model(y, y_predicted)
        rmse_value = np.sqrt(mse_value)

        print("\nModel Evaluation:")
        print(f"  Mean Squared Error (MSE): {mse_value:.2f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse_value:.2f}")
        print(f"  R-squared (R²): {r2_value:.4f}")
        print(f"  (R² close to 1 indicates a good linear fit)")

        # 5. Visualize Results
        plot_regression(X, y, est_intercept, est_slope)

        # 6. Predictive Maintenance Example Usage
        print(f"\nPredictive Example:")
        current_hours = CURRENT_OPERATING_HOURS
        predicted_rul = predict(current_hours, est_intercept, est_slope)
        # Ensure prediction isn't negative
        predicted_rul = max(0, predicted_rul)

        print(f"  For a component with {current_hours} operating hours...")
        print(f"  The estimated RUL is: {predicted_rul:.2f} hours")

        # Maintenance Threshold Check
        maintenance_threshold = MAINTENANCE_THRESHOLD
        if predicted_rul < maintenance_threshold:
            print(f"  ALERT! Estimated RUL ({predicted_rul:.2f}) is below the threshold ({maintenance_threshold} hours).")
            print("  Recommend scheduling maintenance.")
        else:
            print(f"  Estimated RUL ({predicted_rul:.2f}) is above the threshold ({maintenance_threshold} hours).")
            print("  Continue monitoring.")

    except ValueError as ve:
        print(f"\nError during regression calculation: {ve}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")