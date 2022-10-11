"""
ResNet56 calibration experiment on CIFAR10 Dataset
==================================================

We compare multiple calibration algorithms with their reduced, 
class-wise and class-wise reduced counterparts.

We work with a pre-trained ResNet56 classifier trained on the `CIFAR10 Dataset <https://www.cs.toronto.edu/~kriz/cifar.html>`_,
a multi-class classification dataset consisting of 60000 images split evenly across 10 classes

The model achieves an accuracy of roughly 93% of the test set.

Since the model's accuracy is pretty high it is, as expected, well calibrated
(pre-calibration ECE ≈ 0.046, post-calibration ECE <= 0.015).
"""

# %%
# Imports
# -------
import logging
import os
from typing import List

# This import is needed to avoid a circular import error
import kyle.calibration.calibration_methods
import numpy as np
import torch
import torch.nn.functional as F
from kyle.datasets import get_cifar10_dataloader
from kyle.evaluation import EvalStats
from kyle.models.resnet import resnet56
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

from src.constants import DATA_DIR, OUTPUT_DIR, RANDOM_SEED
from src.data_and_models.resnet import download_resnet_models
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
output_dir = OUTPUT_DIR / "resnet56_cifar10"
output_dir.mkdir(exist_ok=True)
output_file = output_dir / "results.csv"

resnet_dir = DATA_DIR / "resnet56_cifar10"
resnet56_model_file = resnet_dir / "resnet56.th"

# %%
n_classes = 10

# %%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# %%
# Configuration
# -------------
set_random_seed(RANDOM_SEED)
configure_plots()

# Required to avoid `RuntimeError: Too many open files.`
# Refer to this issue for for information: https://github.com/pytorch/pytorch/issues/11201
torch.multiprocessing.set_sharing_strategy("file_system")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# %%
# Data
# ----
data_loader = get_cifar10_dataloader(os.fspath(resnet_dir), train=False)

# %%
# Model
# -----
download_resnet_models(resnet_dir)

# %%
model = resnet56()
model = torch.nn.DataParallel(model)
check_point = torch.load(resnet56_model_file, map_location=torch.device("cpu"))
model.load_state_dict(check_point["state_dict"])
model = model.to(device)
model.eval()

# %%
# Evaluating Calibration
# ----------------------
logger.info("Generating model predictions on test set")

logits = []
true_labels = []

with torch.no_grad():
    for features, labels in tqdm(data_loader, total=len(data_loader)):
        features = features.to(device)
        true_labels.append(labels)
        output = model(features)
        output = output.to("cpu")
        logits.append(output)

uncalibrated_confidences = F.softmax(torch.cat(logits), dim=1).numpy()
y_true = torch.cat(true_labels).numpy()

# %%
y_pred = np.argmax(uncalibrated_confidences, axis=1)
model_accuracy = accuracy_score(y_true, y_pred)
logger.info(f"Model accuracy: {model_accuracy*100}%")

# %%
eval_stats = EvalStats(y_true, uncalibrated_confidences, bins=25)
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
    gt_labels=y_true,
    cv=4,
    bins=20,
)

results_df = combined_results_into_dataframe(
    eval_results,
    model_name="ResNet56",
    dataset_name="CIFAR10",
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
    output_file=(output_dir / "evaluation_ECE_resnet56_cifar10.eps"),
    show=False,
)
