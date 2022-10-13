"""
DeiT calibration experiment on RVL-CDIP Dataset
===============================================

We compare multiple calibration algorithms with their reduced, 
class-wise and class-wise reduced counterparts.

We work with a DeiT classifier pre-trained on IIT-CDIP
and then finetuned on `RVL-CDIP <https://adamharley.com/rvl-cdip/>`_, a subset of the former
consisting of 400000 grayscale document images split evenly across 16 classes.

The model achieves an accuracy of roughly 93% of the test set.

Since the model's accuracy is pretty high it is, as expected, well calibrated
(pre-calibration ECE ≈ 0.069, post-calibration ECE <= 0.03).
"""

# %%
# Imports
# -------
import logging
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from kyle.evaluation import EvalStats
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

# %%
# This is needed for notebooks in case jupyter is started directly in the notebooks directory
current_working_directory = Path(".").resolve()
if current_working_directory.name == "notebooks":
    sys.path.insert(0, os.fspath(current_working_directory.parent))

# %%
from src.constants import DATA_DIR, OUTPUT_DIR, RANDOM_SEED
from src.data_and_models.rvl_cdip import download_rvl_cdip
from src.utils import (
    RVLCDIPDataset,
    configure_plots,
    open_image,
    perform_default_evaluation,
    plot_evaluation_results_from_dataframe,
    set_random_seed,
)
from src.utils.evaluation import combined_results_into_dataframe

# %%
# Constants
# -------
output_dir = OUTPUT_DIR / "deit_rvl_cdip"
output_dir.mkdir(exist_ok=True)
output_file = output_dir / "results.csv"

rvl_cdip_dir = DATA_DIR / "deit_rvl_cdip"
images_dir = rvl_cdip_dir / "dataset" / "images"
labels_dir = rvl_cdip_dir / "dataset" / "labels"
features_array_file = rvl_cdip_dir / "dataset" / "features.np"
test_labels_file = labels_dir / "test.txt"
validation_labels_file = labels_dir / "val.txt"
training_labels_file = labels_dir / "train.txt"


# %%
model_name = "microsoft/dit-base-finetuned-rvlcdip"
batch_size = 64
label2idx = {
    "letter": 0,
    "form": 1,
    "email": 2,
    "handwritten": 3,
    "advertisement": 4,
    "scientific_report": 5,
    "scientific_publication": 6,
    "specification": 7,
    "file_folder": 8,
    "news_article": 9,
    "budget": 10,
    "invoice": 11,
    "presentation": 12,
    "questionnaire": 13,
    "resume": 14,
    "memo": 15,
}
idx2label = {v: k for k, v in label2idx.items()}
n_classes = len(label2idx)

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
download_rvl_cdip(rvl_cdip_dir)

# %%
images = []
labels = []

for label_file in [test_labels_file, validation_labels_file]:
    with label_file.open("r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            relative_image_path, label = line.strip().split()
            image_path = images_dir / relative_image_path
            assert image_path.is_file()
            if open_image(image_path) is None:
                continue
            images.append(os.fspath(image_path))
            labels.append(int(label))

feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

image = open_image(images[0])
x = feature_extractor(images=image, return_tensors="np")["pixel_values"][0]

features_array = np.memmap(
    filename=os.fspath(features_array_file),
    shape=(len(labels), *x.shape),
    dtype=float,
    mode="w+",
)

for i, image_file in tqdm(enumerate(images), total=len(images)):
    image = open_image(image_file)
    x = feature_extractor(images=image, return_tensors="np")["pixel_values"][0]
    features_array[i] = x

dataset = RVLCDIPDataset(
    features=features_array,
    labels=labels,
)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
)

# %%
# Model
# -----
model = AutoModelForImageClassification.from_pretrained(model_name)
model = model.float()
model.eval()

# %%
# Evaluating Calibration
# ----------------------
logger.info("Generating model predictions on test set")

uncalibrated_confidences = []
y_true = []

for batch in tqdm(dataloader, total=len(dataloader)):
    features = batch["features"].float()
    labels = batch["label"]
    with torch.no_grad():
        predictions = model(pixel_values=features)
    logits = predictions.logits
    y_true.append(labels.detach().numpy())
    confidences = torch.nn.functional.softmax(logits, dim=1)
    uncalibrated_confidences.append(confidences.detach().numpy())

y_true = np.concatenate(y_true, axis=0)[:, 0]
uncalibrated_confidences = np.concatenate(uncalibrated_confidences, axis=0)

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
)

results_df = combined_results_into_dataframe(
    eval_results,
    model_name="DeiT",
    dataset_name="RVL-CDIP",
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
    output_file=(output_dir / "evaluation_ECE_deit_rvl_cdip.eps"),
    show=False,
)
