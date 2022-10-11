"""
LightGBM calibration experiment on Sorel20M Dataset
===================================================

We compare multiple calibration algorithms with their reduced, 
class-wise and class-wise reduced counterparts.

We work with a pre-trained LightGBM classifier trained on the `SOREL20M Dataset <https://github.com/sophos/SOREL-20M>`_,
a binary classification dataset consisting of nearly
20 million malicious and benign portable executable
files with pre-extracted features and metadata, and high quality labels

The model achieves an accuracy of roughly 98% of the test set.

Since the model's accuracy is pretty high it is, as expected, well calibrated
(pre-calibration ECE ≈ 0.005, post-calibration ECE <= 0.002).
"""

# %%
# Imports
# -------
import logging
import types
from typing import List

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from kyle.evaluation import EvalStats
from sklearn.metrics import accuracy_score

from src.constants import DATA_DIR, OUTPUT_DIR, RANDOM_SEED
from src.data_and_models.sorel20m import download_sorel20m
from src.utils import (
    configure_plots,
    perform_default_evaluation,
    plot_evaluation_results_from_dataframe,
    set_random_seed,
)
from src.utils.evaluation import combined_results_into_dataframe

# %%
# Constants
# -------
output_dir = OUTPUT_DIR / "lightgbm_sorel20m"
output_dir.mkdir(exist_ok=True)
output_file = output_dir / "results.csv"

sorel20m_dir = DATA_DIR / "sorel20m"
lightgbm_model_file = sorel20m_dir / "lightgbm.model"
features_dir = sorel20m_dir / "test-features"
features_file = features_dir / "arr_0.npy" / "arr_0.npy"
labels_file = features_dir / "arr_1.npy" / "arr_1.npy"

n_classes = 2
classes = ["malware", "benign"]

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
download_sorel20m(sorel20m_dir)

# %%
X_test = np.load(features_file, mmap_mode="r")

# Cannot use mmap with netcal because it does stupid things like:
# Checking for array types this way: type(X) != np.ndarray
# Instead of: isinstance(X, np.ndarray), which doesn't work for memory mapped arrays
# because they are a subclass of np.ndarray
# test_labels = np.load(labels_file, mmap_mode="r")
y_test = np.load(labels_file)

# %%
fig, ax = plt.subplots()
values, counts = np.unique(y_test, return_counts=True)
ax.bar(values, counts, edgecolor="k", linewidth=2)
ax.set_xticks(values)
ax.set_xticklabels(classes, rotation=45)

# %%
# Model
# -----
# The following hack is needed because kyle expects classifiers to have the `predict_proba` method
# and to output an array of the same dimensionality as the number of classes
def predict_proba(self, X):
    y = model.predict(X)
    y = y[:, np.newaxis]
    y = np.append(
        np.zeros(y.shape, dtype=float),
        y,
        axis=1,
    )
    y[:, 0] = 1.0 - y[:, 1]
    return y


model = lgb.Booster(model_file=lightgbm_model_file)
model.predict_proba = types.MethodType(predict_proba, model)

# %%
lgb.plot_tree(model)

# %%
# Evaluating Calibration
# ----------------------
uncalibrated_confidences = model.predict_proba(X_test)

# %%
y_pred = np.argmax(uncalibrated_confidences, axis=1)
model_accuracy = accuracy_score(y_test, y_pred)
logger.info(f"Model accuracy: {model_accuracy*100}%")

# %%
eval_stats = EvalStats(y_test, uncalibrated_confidences, bins=25)
logger.info(f"ECE before calibration: {eval_stats.expected_calibration_error()}")

# %%
# Recalibration
# -------------
# We evaluate reduction wrappers on multiple metrics with different calibration algorithms

# %%
# Balanced
# ~~~~~~~~
logger.info("Performing evaluation")

eval_results = perform_default_evaluation(
    confidences=uncalibrated_confidences,
    gt_labels=y_test,
)

results_df = combined_results_into_dataframe(
    eval_results,
    model_name="LightGBM",
    dataset_name="SOREL20M",
)

# %%
reduction_methods_order: List[str] = results_df["Reduction Method"].unique().tolist()
reduction_methods_order = [reduction_methods_order[0]] + sorted(
    reduction_methods_order[1:], key=len
)

# %%
# Save Results
# ------------
logger.info("Saving results")
results_df.to_csv(output_file, sep=";", index=False)

# %%
# Plots
# -----
logger.info("Plotting results")

results_df = results_df.query("(Metric != 'condition') & (Metric != 'weak_condition')")

plot_evaluation_results_from_dataframe(
    results_df,
    hue_order=reduction_methods_order,
    output_file=(output_dir / "evaluation_ECE_lightgbm_sorel20m.eps"),
    show=False,
)
