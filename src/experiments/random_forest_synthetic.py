"""
Random Forest calibration experiment on Synthetic Data
======================================================

We compare multiple calibration algorithms with their reduced, 
class-wise and class-wise reduced counterparts.

We work with a random forest classifier trained on two synthetic datasets each containing
5 classes and 60k samples, where one of them is imbalanced.

The model is trained on 30k samples (from a stratified shuffle split)
and achieves an accuracy of roughly 89% in both cases.

As is common with random forests, the resulting model is highly miscalibrated
(pre-calibration ECE ≈ 0.23, post-calibration ECE < 0.03).
"""

# %%
# Imports
# -------
import logging
import os
import sys
from pathlib import Path
from typing import List

import pandas as pd
from kyle.evaluation import EvalStats

# %%
# This is needed for notebooks in case jupyter is started directly in the notebooks directory
current_working_directory = Path(".").resolve()
if current_working_directory.name == "notebooks":
    sys.path.insert(0, os.fspath(current_working_directory.parent))

# %%
from src.constants import OUTPUT_DIR, RANDOM_SEED
from src.utils import (
    configure_plots,
    perform_default_evaluation,
    plot_evaluation_results_from_dataframe,
    set_random_seed,
)
from src.utils.evaluation import combined_results_into_dataframe
from src.utils.other import get_rf_calibration_dataset

# %%
# Constants
# -------
output_dir = OUTPUT_DIR / "random_forest_synthetic"
output_dir.mkdir(exist_ok=True)

IMBALANCED_WEIGHTS = (0.3, 0.1, 0.25, 0.15)

# %%
# Configuration
# -------------
set_random_seed(RANDOM_SEED)
configure_plots()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# %%
# Data and Models
# ---------------
logger.info(f"Creating balanced dataset")
balanced_confs, balanced_gt = get_rf_calibration_dataset()

logger.info(f"Creating imbalanced dataset")
imbalanced_confs, imbalanced_gt = get_rf_calibration_dataset(weights=IMBALANCED_WEIGHTS)

# %%
# Evaluating Calibration
# ----------------------
eval_stats = EvalStats(balanced_gt, balanced_confs, bins=25)
logger.info(
    f"Balanced ECE before calibration: {eval_stats.expected_calibration_error()}"
)

eval_stats = EvalStats(imbalanced_gt, imbalanced_confs, bins=25)
logger.info(
    f"Imbalanced ECE before calibration: {eval_stats.expected_calibration_error()}"
)
# %%
# Recalibration
# -------------
# We evaluate reduction wrappers on multiple metrics with different calibration algorithms

# %%
# Balanced
# ~~~~~~~~
logger.info("Performing evaluation for balanced dataset")

balanced_eval_results = perform_default_evaluation(
    confidences=balanced_confs,
    gt_labels=balanced_gt,
)

rf_balanced_results_df = combined_results_into_dataframe(
    balanced_eval_results,
    model_name="Random Forest",
    dataset_name="Synthetic Balanced",
)

# %%
reduction_methods_order: List[str] = (
    rf_balanced_results_df["Reduction Method"].unique().tolist()
)
reduction_methods_order = [reduction_methods_order[0]] + sorted(
    reduction_methods_order[1:], key=len
)

# %%
# Imbalanced
# ~~~~~~~~~~
logger.info("Performing evaluation for imbalanced dataset")

imbalanced_eval_results = perform_default_evaluation(
    confidences=imbalanced_confs,
    gt_labels=imbalanced_gt,
)

rf_imbalanced_results_df = combined_results_into_dataframe(
    imbalanced_eval_results,
    model_name="Random Forest",
    dataset_name="Synthetic Imbalanced",
)

# %%
# Save Results
# ------------
results_df = pd.concat([rf_balanced_results_df, rf_imbalanced_results_df])
logger.info("Saving results")
output_file = output_dir / "results.csv"
results_df.to_csv(output_file, sep=";", index=False)

# %%
# Plots
# -----
logger.info("Plotting results")

results_df = results_df.query("(Metric != 'condition') & (Metric != 'weak_condition')")

plot_evaluation_results_from_dataframe(
    results_df.query(
        "(Dataset == 'Synthetic Balanced') "
        "& (`Calibration Method` != 'TemperatureScaling') "
    ),
    hue_order=reduction_methods_order,
    output_file=(output_dir / "evaluation_ECE_rf_balanced.eps"),
    show=False,
)

plot_evaluation_results_from_dataframe(
    results_df.query(
        "(Dataset == 'Synthetic Imbalanced') "
        "& (`Calibration Method` != 'TemperatureScaling') "
    ),
    hue_order=reduction_methods_order,
    output_file=(output_dir / "evaluation_ECE_rf_balanced.eps"),
    show=False,
)

plot_evaluation_results_from_dataframe(
    results_df.query("(`Calibration Method` == 'TemperatureScaling') "),
    hue_order=reduction_methods_order,
    output_file=(output_dir / "evaluation_ECE_rf_temperature_scaling.eps"),
    show=False,
)
