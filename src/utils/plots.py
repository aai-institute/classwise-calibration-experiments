import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src.constants import DEFAULT_BINS

from .other import is_notebook

__all__ = [
    "configure_plots",
    "plot_scores",
    "plot_variation_of_confidences",
    "plot_default_evaluation_results",
    "plot_evaluation_results_from_dataframe",
]


def configure_plots() -> None:
    sns.set_theme(style="whitegrid", palette="pastel")
    plt.rcParams["figure.figsize"] = (20, 14)
    plt.rcParams["font.size"] = 15
    plt.rcParams["xtick.labelsize"] = 15
    plt.rcParams["ytick.labelsize"] = 15


def plot_scores(
    wrapper_scores_dict: dict, title="", ax=None, y_lim=None, *, rotate_labels: int = 45
):
    labels = wrapper_scores_dict.keys()
    scores_collection = wrapper_scores_dict.values()

    if ax is None:
        _, ax = plt.subplots()
    ax.set_title(title)
    ax.boxplot(scores_collection, labels=labels)
    for tick in ax.get_xticklabels():
        tick.set_rotation(rotate_labels)
    if y_lim is not None:
        ax.set_ylim(y_lim)


def plot_evaluation_results_from_dataframe(
    df: pd.DataFrame,
    *,
    hue_order: Optional[List[str]] = None,
    show: bool = True,
    output_file: Optional[Path] = None,
):
    g = sns.catplot(
        x="Calibration Method",
        y="Score",
        hue="Reduction Method",
        col="Metric",
        kind="box",
        sharey=False,
        sharex=False,
        col_wrap=1,
        data=df,
        height=8.27,
        aspect=11.7 / 8.27,
        width=0.6,
        hue_order=hue_order,
    )
    for ax in g.axes:
        ax.set_ylabel("")

    sns.move_legend(
        g,
        "upper left",
        bbox_to_anchor=(0.15, 1.0),
        fontsize="large",
        title_fontsize="large",
        ncol=4,
        edgecolor="black",
        frameon=True,
    )
    plt.tight_layout()
    plt.gcf().subplots_adjust(top=0.90)
    if output_file is not None:
        plt.savefig(
            os.fspath(output_file.with_suffix(".eps")),
            format="eps",
            bbox_inches="tight",
            pad_inches=0,
        )

    if is_notebook() or show:
        plt.show()


def plot_default_evaluation_results(
    evaluation_results: dict,
    title_addon=None,
    *,
    sharey: bool = True,
    rotate_labels: int = 45,
    show: bool = True,
    output_dir: Optional[Path] = None,
):
    ncols = len(list(evaluation_results.values())[0])
    for metric, results in evaluation_results.items():
        fig, axes = plt.subplots(nrows=1, ncols=ncols, sharey=sharey)
        if ncols == 1:  # axes fails to be a list if ncols=1
            axes = [axes]
        for col, result in zip(axes, results):
            wrapper_scores_dict, description = result
            plot_scores(
                wrapper_scores_dict,
                title=description,
                ax=col,
                rotate_labels=rotate_labels,
            )

        title = f"Evaluation with {metric} - {DEFAULT_BINS} bins"
        if title_addon is not None:
            title += f"\n{title_addon}"
        fig.suptitle(title)
        fig.tight_layout()

        if output_dir is not None:
            output_file = output_dir / title.lower().replace(" ", "_")
            fig.savefig(
                output_file.with_suffix(".eps"),
                format="eps",
                bbox_inches="tight",
                pad_inches=0,
            )

        if show:
            plt.show()


def plot_variation_of_confidences(confidences, *, ax=None, n_bins: int = 10):
    hists = []
    for i in range(confidences.shape[1]):
        hist, _ = np.histogram(confidences[:, i], bins=n_bins)
        hists.append(hist)
    hists = np.stack(hists)
    means = np.mean(hists, axis=0)
    errors = np.std(hists, axis=0)
    if ax is None:
        _, ax = plt.subplots()
    x_values = np.arange(n_bins)
    ax.bar(x_values, means, edgecolor="k", linewidth=2)
    ax.errorbar(x_values, means, yerr=errors, fmt="o", color="y")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    return ax
