"""
Script for combining the results of the calibration experiments
===================================================================

We combine the different results of the calibration experiments and create the plots
that will be used to present the results.
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

# %%
# This is needed for notebooks in case jupyter is started directly in the notebooks directory
current_working_directory = Path(".").resolve()
if current_working_directory.name == "notebooks":
    sys.path.insert(0, os.fspath(current_working_directory.parent))

# %%
from src.constants import OUTPUT_DIR, RANDOM_SEED
from src.utils import (
    configure_plots,
    create_summary_table_with_absolute_values_and_stddev,
    create_summary_table_with_relative_change_and_stddev,
    plot_evaluation_results_from_dataframe,
    set_random_seed,
)

# %%
# Constants
# -------
output_dir = OUTPUT_DIR / "combined_results"
output_dir.mkdir(exist_ok=True)

# %%
# Configuration
# -------------
set_random_seed(RANDOM_SEED)
configure_plots()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


# %%
# Data
# ----

all_results = []

for result_file in OUTPUT_DIR.rglob("results.csv"):
    df = pd.read_csv(result_file, sep=";")
    all_results.append(df)

results_df = pd.concat(all_results)

# %%
# We filter the results first
results_df = results_df.query("(Metric != 'condition') & (Metric != 'weak_condition')")
results_df = results_df[
    ~results_df["Reduction Method"].str.lower().str.contains("weighted")
]

# %%
# Plots
# -----
reduction_methods_order: List[str] = results_df["Reduction Method"].unique().tolist()
reduction_methods_order = [reduction_methods_order[0]] + sorted(
    reduction_methods_order[1:], key=len
)

# %%
# Random Forest
# ~~~~~~~~~~~~~

# %%
# Balanced Dataset
# ++++++++++++++++

logger.info(
    "Comparison of Beta calibration, isotonic regression and histogram binning "
    "for a random forest model trained on a balanced synthetic dataset (6 folds; 25 bins). "
    "For each method we plot the cross-validated estimate of ECE / cwECE together with its standard error for the baseline and class-wise, "
    "reduced and class-wise reduced variants."
)

output_file = output_dir / "evaluation_ECE_rf_balanced.eps"

plot_evaluation_results_from_dataframe(
    df=results_df.query(
        "(Model == 'Random Forest') "
        "& (Dataset == 'Synthetic Balanced') "
        "& (`Calibration Method` != 'TemperatureScaling') "
    ),
    hue_order=reduction_methods_order,
    output_file=output_file,
    show=False,
)

# %%
# Imbalanced Dataset
# ++++++++++++++++++

logger.info(
    "Comparison of Beta calibration, isotonic regression and histogram binning with different Reduction methods "
    "for a random forest model trained on an imbalanced synthetic dataset (6 folds; 25 bins)"
)

output_file = output_dir / "evaluation_ECE_rf_imbalanced.eps"

plot_evaluation_results_from_dataframe(
    df=results_df.query(
        "(Model == 'Random Forest') "
        "& (Dataset == 'Synthetic Imbalanced') "
        "& (`Calibration Method` != 'TemperatureScaling') "
    ),
    hue_order=reduction_methods_order,
    output_file=output_file,
    show=False,
)

# %%
# Temperature Scaling and Random Forest
# +++++++++++++++++++++++++++++++++++++

logger.info(
    "Evaluation of Temperature Scaling for a random forest model trained on synthetic datasets (6 folds; 25 bins). "
    "For this specific problem, the distribution of the confidences hinders the method from correctly recalibrating, "
    "independently of the reduction."
)

output_file = output_dir / "evaluation_ECE_rf_temperature_scaling.eps"

plot_evaluation_results_from_dataframe(
    df=results_df.query(
        "(Model == 'Random Forest') "
        "& (`Calibration Method` == 'TemperatureScaling') "
    ),
    hue_order=reduction_methods_order,
    output_file=output_file,
    show=False,
)

# %%
# Other Models and Datasets
# ~~~~~~~~~~~~~~~~~~~~~~~~~

# %%
# ECE
# +++

logger.info(
    "Comparison of Beta calibration, isotonic regression, histogram binning and temperature scaling "
    "for 3 models trained on 3 real world datasets (6 folds; 25 bins). "
    "For each method we plot the cross-validated estimate of ECE together with its standard error "
    "for the baseline and class-wise, reduced and class-wise reduced variants."
)

output_file = output_dir / "evaluation_ECE_multiple.eps"

plot_evaluation_results_from_dataframe(
    df=results_df.query("(Model != 'Random Forest') & (Metric == 'ECE')"),
    hue_order=reduction_methods_order,
    output_file=output_file,
    show=False,
)

# %%
# cwECE
# +++++

logger.info(
    "Comparison of Beta calibration, isotonic regression, histogram binning and temperature scaling "
    "for 3 models trained on 3 real world datasets (6 folds; 25 bins). "
    "For each method we plot the cross-validated estimate of cwECE together with its standard error "
    "for the baseline and class-wise, reduced and class-wise reduced variants."
)

output_file = output_dir / "evaluation_cwECE_multiple.eps"

plot_evaluation_results_from_dataframe(
    df=results_df.query("(Model != 'Random Forest') & (Metric == 'cwECE')"),
    hue_order=reduction_methods_order,
    output_file=output_file,
    show=False,
)

# %%
# Tables
# ------

# %%
# ECE
# ~~~

ece_summary_df = create_summary_table_with_absolute_values_and_stddev(
    results_df[results_df["Metric"] == "ECE"], reduction_methods_order
)
ece_summary_df.to_csv(output_dir / "ece_results_summary_absolute.csv")

ece_summary_df_relative_change = create_summary_table_with_relative_change_and_stddev(
    results_df[results_df["Metric"] == "ECE"].copy(),
    reduction_methods_order,
)
ece_summary_df_relative_change.to_csv(output_dir / "ece_results_summary_relative.csv")

# %%
# cwECE
# ~~~

cwece_summary_df = create_summary_table_with_absolute_values_and_stddev(
    results_df[results_df["Metric"] == "cwECE"], reduction_methods_order
)
cwece_summary_df.to_csv(output_dir / "cwece_results_summary_absolute.csv")

cwece_summary_df_relative_change = create_summary_table_with_relative_change_and_stddev(
    results_df[results_df["Metric"] == "cwECE"].copy(),
    reduction_methods_order,
)
cwece_summary_df_relative_change.to_csv(
    output_dir / "cwece_results_summary_relative.csv"
)
