import numpy as np
from matplotlib import pyplot as plt

from src.constants import DEFAULT_BINS

__all__ = [
    "plot_scores",
    "plot_variation_of_confidences",
    "plot_default_evaluation_results",
]


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


def plot_default_evaluation_results(
    evaluation_results: dict,
    title_addon=None,
    *,
    sharey: bool = True,
    rotate_labels: int = 45,
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

        title = f"Evaluation with {metric}; {DEFAULT_BINS} bins)"
        if title_addon is not None:
            title += f"\n{title_addon}"
        fig.suptitle(title)
        plt.tight_layout()
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
