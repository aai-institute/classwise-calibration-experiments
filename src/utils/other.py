import logging
import random
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

from src.constants import RANDOM_SEED

__all__ = ["set_random_seed"]

logger = logging.getLogger(__name__)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def is_notebook() -> bool:
    """Taken verbatim from here:
    https://stackoverflow.com/a/39662359
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def get_rf_calibration_dataset(
    n_classes: int = 5,
    weights: Optional[Tuple[float, ...]] = None,
    n_samples: int = 60000,
    n_informative: int = 15,
    random_seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    n_dataset_samples = 2 * n_samples
    test_size = 0.5
    X, y = make_classification(
        n_samples=n_dataset_samples,
        n_classes=n_classes,
        n_informative=n_informative,
        weights=weights,
        random_state=random_seed,
    )
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_seed
    )

    train_index, test_index = list(sss.split(X, y))[0]
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]

    model = RandomForestClassifier(random_state=random_seed)
    model.fit(X_train, y_train)
    confidences = model.predict_proba(X_test)
    y_pred = confidences.argmax(1)
    accuracy = accuracy_score(y_pred, y_test)
    logger.info(f"Model accuracy: {accuracy}")
    return confidences, y_test
