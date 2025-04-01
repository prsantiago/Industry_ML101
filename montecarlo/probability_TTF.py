#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Monte Carlo simulation for estimating conditional failure probability
of a component following a Weibull distribution for its Time To Failure (TTF).
Includes saving a plot of the simulated TTF distribution.
"""

import random
import math
import os
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt

# --- Simulation Configuration Constants ---
WEIBULL_SHAPE: float = 2.5
WEIBULL_SCALE: float = 5000.0  # Characteristic life in hours
CURRENT_AGE_HOURS: float = 4000.0 # Component age in hours
PREDICTION_HORIZON_HOURS: float = 500.0 # Future window in hours
NUM_SIMULATIONS: int = 100000 # Number of Monte Carlo trials
RISK_THRESHOLD: float = 0.10 # 10% risk threshold for maintenance alert

# --- Plotting Configuration ---
# Directory relative to the script location
OUTPUT_PLOT_DIR: str = "montecarlo"
# Filename for the output plot
OUTPUT_PLOT_FILENAME: str = "./assests/plots/montecarlo_simulation_results.png"
# --- End of Configuration Constants ---


def generate_weibull_ttf(shape: float, scale: float) -> float:
    """
    Generates a simulated Time To Failure (TTF)
    from a Weibull distribution using the inverse transform sampling method.

    Args:
        shape: Shape parameter (beta, β) of the Weibull distribution. Must be > 0.
        scale: Scale parameter (eta, η) of the Weibull distribution. Must be > 0.

    Returns:
        A simulated time to failure in hours.
    """
    if shape <= 0 or scale <= 0:
        raise ValueError("Weibull shape and scale parameters must be positive.")
    u = random.uniform(1e-10, 1.0)
    ttf = scale * (-math.log(u))**(1.0 / shape)
    return ttf


def estimate_conditional_failure_probability(
    shape: float,
    scale: float,
    current_age_hours: float,
    prediction_horizon_hours: float,
    num_simulations: int = 10000
) -> Tuple[float, List[float]]:
    """
    Estimates the CONDITIONAL probability that a component will fail within
    a future time horizon, GIVEN that it has survived up to its current age.
    Also returns the list of all simulated TTFs.

    Args:
        shape: Weibull shape parameter of the component.
        scale: Weibull scale parameter of the component.
        current_age_hours: Accumulated operating hours of the component so far.
        prediction_horizon_hours: Future time window (in hours) for which
                                  the failure probability is estimated.
        num_simulations: Number of Monte Carlo simulations to run.

    Returns:
        A tuple containing:
            - The estimated conditional probability of failure (float, NaN if undefined).
            - A list of all simulated TTF values (List[float]).
    """
    if current_age_hours < 0 or prediction_horizon_hours < 0 or num_simulations <= 0:
        raise ValueError("Age, horizon, and number of simulations must be non-negative (simulations > 0).")

    survivors_up_to_current_age = 0
    failures_within_horizon_conditional = 0
    all_simulated_ttfs: List[float] = []

    for _ in range(num_simulations):
        total_simulated_ttf = generate_weibull_ttf(shape, scale)
        all_simulated_ttfs.append(total_simulated_ttf)

        if total_simulated_ttf > current_age_hours:
            survivors_up_to_current_age += 1
            if total_simulated_ttf <= (current_age_hours + prediction_horizon_hours):
                failures_within_horizon_conditional += 1

    if survivors_up_to_current_age == 0:
        print("\nWarning: No simulation survived up to the current age.")
        failure_probability = float('nan')
    else:
        failure_probability = failures_within_horizon_conditional / survivors_up_to_current_age

    return failure_probability, all_simulated_ttfs


def plot_and_save_ttf_histogram(
    all_simulated_ttfs: List[float],
    shape: float,
    scale: float,
    current_age_hours: float,
    prediction_horizon_hours: float,
    failure_prob: float,
    plot_dir: str,
    plot_filename: str
) -> None:
    """
    Generates and saves a histogram of the simulated TTF values to a specified path.

    Args:
        all_simulated_ttfs: List of simulated Time To Failure values.
        shape: Weibull shape parameter used in simulation.
        scale: Weibull scale parameter used in simulation.
        current_age_hours: Current age of the component (for plotting lines).
        prediction_horizon_hours: Prediction horizon (for plotting lines).
        failure_prob: The calculated conditional failure probability.
        plot_dir: The directory where the plot will be saved.
        plot_filename: The filename for the saved plot.
    """
    if not all_simulated_ttfs:
        print("Warning: No simulation data to plot.")
        return

    # Construct the full output path
    full_output_path = os.path.join(plot_dir, plot_filename)

    # Create the output directory if it doesn't exist
    try:
        os.makedirs(plot_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {plot_dir}: {e}")
        # Optionally decide if you want to proceed without saving the plot
        return

    plt.figure(figsize=(12, 7))
    plt.hist(all_simulated_ttfs, bins=50, density=True, alpha=0.7, label='Simulated TTF Distribution', color='skyblue', edgecolor='black')
    ymin, ymax = plt.ylim()
    plt.axvline(current_age_hours, color='red', linestyle='--', linewidth=2,
                label=f'Current Age ({current_age_hours:,.0f} hrs)')
    end_horizon_time = current_age_hours + prediction_horizon_hours
    plt.axvline(end_horizon_time, color='orange', linestyle=':', linewidth=2,
                label=f'End of Horizon ({end_horizon_time:,.0f} hrs)')
    plt.fill_betweenx([ymin, ymax], current_age_hours, end_horizon_time,
                      color='red', alpha=0.1, label='Prediction Horizon')
    plt.ylim(ymin, ymax)
    plt.title(f'Simulated Weibull TTF Distribution (Shape={shape}, Scale={scale})', fontsize=16)
    plt.xlabel('Simulated Time To Failure (Hours)', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)

    if not math.isnan(failure_prob):
        prob_text = (f'Est. Cond. Probability:\n'
                     f'P(Fail in next {prediction_horizon_hours:,.0f} hrs | Survived {current_age_hours:,.0f} hrs)\n'
                     f'= {failure_prob:.2%}')
        plt.text(0.97, 0.97, prob_text, transform=plt.gca().transAxes,
                 horizontalalignment='right', verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.8),
                 fontsize=10)

    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Always save the plot
    try:
        plt.savefig(full_output_path, dpi=300, bbox_inches='tight') # Use bbox_inches for potentially better layout saving
        print(f"\nPlot saved to {full_output_path}")
    except Exception as e:
        print(f"\nError saving plot to {full_output_path}: {e}")
    finally:
        # Close the plot figure to free up memory, especially if running many times
        plt.close()


def main():
    """
    Main function to run simulation using constants, print results, and save plot.
    """
    # --- Run the Simulation ---
    print("Starting Monte Carlo simulation...")
    try:
        estimated_prob, all_ttfs = estimate_conditional_failure_probability(
            WEIBULL_SHAPE,
            WEIBULL_SCALE,
            CURRENT_AGE_HOURS,
            PREDICTION_HORIZON_HOURS,
            NUM_SIMULATIONS
        )
    except ValueError as e:
        print(f"Error in simulation parameters: {e}")
        return

    # --- Display Results ---
    print("\n--- Simulation Parameters Used ---")
    print(f"Weibull Shape (β): {WEIBULL_SHAPE}")
    print(f"Weibull Scale (η): {WEIBULL_SCALE:,.1f} hours")
    print(f"Component Current Age: {CURRENT_AGE_HOURS:,.1f} hours")
    print(f"Prediction Horizon: {PREDICTION_HORIZON_HOURS:,.1f} hours")
    print(f"Number of Simulations: {NUM_SIMULATIONS:,}")

    print("\n--- Simulation Results ---")
    if not math.isnan(estimated_prob):
        print(f"Estimated CONDITIONAL probability of failure in the next {PREDICTION_HORIZON_HOURS:,.0f} hours: "
              f"{estimated_prob:.4f} (or {estimated_prob*100:.2f}%)")

        # --- Example Decision Logic ---
        if estimated_prob > RISK_THRESHOLD:
            print(f"\nALERT! Failure probability ({estimated_prob*100:.2f}%) exceeds the threshold "
                  f"({RISK_THRESHOLD*100:.1f}%). Maintenance recommended.")
        else:
            print(f"\nFailure probability ({estimated_prob*100:.2f}%) is below the threshold "
                  f"({RISK_THRESHOLD*100:.1f}%). Continue monitoring.")
    else:
        print("Could not calculate conditional failure probability (check parameters and simulation count).")

    # --- Generate and Save Plot ---
    plot_and_save_ttf_histogram(
        all_ttfs,
        WEIBULL_SHAPE,
        WEIBULL_SCALE,
        CURRENT_AGE_HOURS,
        PREDICTION_HORIZON_HOURS,
        estimated_prob,
        OUTPUT_PLOT_DIR,
        OUTPUT_PLOT_FILENAME
    )
    print("\nSimulation finished.")


# Standard Python entry point
if __name__ == "__main__":
    main()