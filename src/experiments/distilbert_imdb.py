"""
Script for the DistilBERT calibration experiment on IMDB Dataset
================================================================

We compare multiple calibration algorithms with their reduced, 
class-wise and class-wise reduced counterparts. We use for this experiment 4 cross-validation splits
and 20 bins for ECE and cwECE instead of 5 and 25, respectively, like for the other experiments
because of the low number of samples in the dataset.

We work with a pre-trained DistilBERT classifier trained by distilling the
BERT base model and then fine-tuned on the `IMDB Dataset <http://ai.stanford.edu/~amaas/data/sentiment/>`_,
a dataset for binary sentiment classification consisting of 50000 movie reviews.

The model achieves an accuracy of roughly 92% of the test set.

Since the model's accuracy is pretty high it is, as expected, well calibrated
(pre-calibration ECE ≈ 0.043, post-calibration ECE <= 0.02).
"""

# %%
# Imports
# -------
import logging
import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torchtext
from kyle.evaluation import EvalStats
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

from src.constants import DATA_DIR, OUTPUT_DIR, RANDOM_SEED
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
output_dir = OUTPUT_DIR / "distilbert_imdb"
output_dir.mkdir(exist_ok=True)
output_file = output_dir / "results.csv"

imdb_dir = DATA_DIR / "distilbert_imdb"


# %%
model_name = "textattack/distilbert-base-uncased-imdb"
batch_size = 64
classes = ["neg", "pos"]
n_classes = len(classes)

# %%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
datapipe = torchtext.datasets.IMDB(root=os.fspath(imdb_dir), split="test")
datapipe = datapipe.batch(batch_size).rows2columnar(["label", "text"])
datapipe = datapipe.map(
    lambda x: [
        tokenizer(x["text"], return_tensors="pt", truncation=True, padding=True),
        torch.LongTensor(list(map(lambda k: classes.index(k), x["label"]))),
    ]
)
dataloader = DataLoader(datapipe, batch_size=None)

# %%
# Model
# -----
model = DistilBertForSequenceClassification.from_pretrained(model_name)
model = model.to(device)
model.eval()

# %%
# Evaluating Calibration
# ----------------------
logger.info("Generating model predictions on test set")

all_logits = []
y_true = []

with torch.no_grad():
    for inputs, labels in tqdm(dataloader, total=25000 // batch_size):
        y_true.append(labels)
        inputs = inputs.to(device)
        output = model(**inputs)
        logits = output.logits
        logits = logits.to("cpu")
        all_logits.append(logits)

all_logits = torch.cat(all_logits)
uncalibrated_confidences = F.softmax(all_logits, dim=1).numpy()
y_true = torch.cat(y_true).numpy()

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
    model_name="DistilBERT",
    dataset_name="IMDB",
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
    output_file=(output_dir / "evaluation_ECE_distilbert_imdb.eps"),
    show=False,
)
