import contextlib
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from kyle.calibration.calibration_methods import (
    BetaCalibration,
    ClassWiseCalibration,
    IsotonicRegression,
    TemperatureScaling,
)
from kyle.evaluation import EvalStats
from sklearn.model_selection import cross_val_score
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.constants import ALL_METRICS, DEFAULT_BINS, DEFAULT_CV
from src.utils.calibration import (
    ConfidenceReducedCalibration,
    HistogramBinning,
    WeightedConfidenceReducedCalibration,
)

logger = logging.getLogger(__name__)


__all__ = [
    "evaluate_calibration_wrappers",
    "perform_default_evaluation",
    "perform_corollary_condition_evaluation",
]

ALL_CALIBRATION_METHOD_FACTORIES = (
    TemperatureScaling,
    BetaCalibration,
    IsotonicRegression,
    HistogramBinning,
)
DEFAULT_WRAPPERS = {
    "Baseline": lambda method_factory: method_factory(),
    "Reduced": lambda method_factory: ConfidenceReducedCalibration(method_factory()),
    "Class-wise": lambda method_factory: ClassWiseCalibration(method_factory),
    "Class-wise reduced": lambda method_factory: ClassWiseCalibration(
        lambda: ConfidenceReducedCalibration(method_factory())
    ),
}

ALL_WRAPPERS = {
    **DEFAULT_WRAPPERS,
    "Weighted Reduced": lambda method_factory: WeightedConfidenceReducedCalibration(
        method_factory()
    ),
    "Class-wise weighted reduced": lambda method_factory: ClassWiseCalibration(
        lambda: WeightedConfidenceReducedCalibration(method_factory())
    ),
}

WRAPPER_FOR_CHECKING_CONDITION = {
    "Reduced": lambda method_factory: ConfidenceReducedCalibration(method_factory()),
    "Weighted Reduced": lambda method_factory: WeightedConfidenceReducedCalibration(
        method_factory()
    ),
}


def compute_score(scaler, confs: np.ndarray, labels: np.ndarray, bins, metric="ECE"):
    calibrated_confs = scaler.get_calibrated_confidences(confs)
    eval_stats = EvalStats(labels, calibrated_confs, bins=bins)
    if metric == "ECE":
        score = eval_stats.expected_calibration_error()
    elif metric == "cwECE":
        score = eval_stats.class_wise_expected_calibration_error()
    elif isinstance(metric, int):
        score = eval_stats.expected_marginal_calibration_error(metric)
    elif metric == "condition":
        if hasattr(scaler, "satisfies_condition_percentage"):
            result = scaler.satisfies_condition_percentage(confs)
        score = result
    else:
        raise ValueError(f"Unknown metric {metric}")
    return score


def get_scores(scaler, metric, cv, bins, confs, labels):
    scoring = lambda *args: compute_score(*args, bins=bins, metric=metric)
    # Ugly HACK to prevent NetCal from printing directly to stdout
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        return cross_val_score(
            scaler,
            confs,
            labels,
            scoring=scoring,
            cv=cv,
            error_score="raise",
            verbose=False,
        )


def evaluate_calibration_wrappers(
    method_factory,
    confidences,
    gt_labels,
    wrappers_dict=None,
    metric="ECE",
    cv=DEFAULT_CV,
    method_name=None,
    bins=DEFAULT_BINS,
    short_description=False,
):
    if method_name is None:
        method_name = method_factory.__name__
    if short_description:
        description = f"{method_name}"
    else:
        description = (
            f"Evaluating wrappers of {method_name} on metric {metric} with {bins} bins\n "
            f"CV with {cv} folds on {len(confidences)} data points."
        )
    if wrappers_dict is None:
        wrappers_dict = DEFAULT_WRAPPERS

    wrapper_scores_dict = {}
    for wrapper_name, wrapper in wrappers_dict.items():
        if metric == "weak_condition" and wrapper_name != "Weighted Reduced":
            continue
        method = wrapper(method_factory)
        scores = get_scores(
            method, metric, cv=cv, bins=bins, confs=confidences, labels=gt_labels
        )
        wrapper_scores_dict[wrapper_name] = scores
    return wrapper_scores_dict, description


def perform_default_evaluation(
    confidences,
    gt_labels,
    method_factories=ALL_CALIBRATION_METHOD_FACTORIES,
    metrics=ALL_METRICS,
    cv=DEFAULT_CV,
    wrappers_dict=DEFAULT_WRAPPERS,
    bins=DEFAULT_BINS,
):
    evaluation_results = defaultdict(list)
    with logging_redirect_tqdm():
        for metric in tqdm(metrics, desc="Calibration Metrics"):
            logger.info(f"Creating evaluation for {metric}")
            for method_factory in tqdm(method_factories, desc="Reduction Methods"):
                logger.info(f"Computing scores for {method_factory.__name__}")
                result = evaluate_calibration_wrappers(
                    method_factory,
                    confidences=confidences,
                    gt_labels=gt_labels,
                    metric=metric,
                    short_description=True,
                    cv=cv,
                    wrappers_dict=wrappers_dict,
                    bins=bins,
                )
                evaluation_results[metric].append(result)
    return evaluation_results


def perform_corollary_condition_evaluation(
    confidences,
    gt_labels,
    method_factories=ALL_CALIBRATION_METHOD_FACTORIES,
    cv=DEFAULT_CV,
    wrappers_dict=WRAPPER_FOR_CHECKING_CONDITION,
    bins=DEFAULT_BINS,
):
    results = {"condition": []}
    logger.info("Creating evaluation for corollary conditions")
    with logging_redirect_tqdm():
        for method_factory in method_factories:
            logger.info(f"Computing scores for {method_factory.__name__}")
            result = evaluate_calibration_wrappers(
                method_factory,
                confidences=confidences,
                gt_labels=gt_labels,
                metric="condition",
                short_description=True,
                cv=cv,
                wrappers_dict=wrappers_dict,
                bins=bins,
            )
            results["condition"].append(result)
    return results


def combined_results_into_dataframe(
    eval_results: dict, *, model_name: str, dataset_name: str
) -> pd.DataFrame:
    dataframes = []

    for metric, values in eval_results.items():
        df = pd.DataFrame(values, columns=["values", "Calibration Method"])
        df = pd.concat([df, df["values"].apply(pd.Series)], axis=1)
        df = df.drop(columns=["values"])
        value_vars = [x for x in df.columns if x != "Calibration Method"]
        df = df.melt(
            id_vars=["Calibration Method"],
            value_vars=value_vars,
            var_name="Reduction Method",
            value_name="Score",
        )
        df = df.explode("Score")
        df["Metric"] = metric
        dataframes.append(df)

    results_df = pd.concat(dataframes)
    results_df["Model"] = model_name
    results_df["Dataset"] = dataset_name
    results_df["Model, Dataset"] = results_df["Model"] + ", " + results_df["Dataset"]
    return results_df
