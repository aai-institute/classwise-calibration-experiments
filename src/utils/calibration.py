import numpy as np
from kyle.calibration.calibration_methods import (
    ConfidenceReducedCalibration,
    get_reduced_confidences,
    HistogramBinning,
    TemperatureScaling,
    BetaCalibration,
    IsotonicRegression,
    ClassWiseCalibration,
)


class ConfidenceReducedCalibration(ConfidenceReducedCalibration):
    def satisfies_condition_percentage(self, confidences: np.ndarray):
        """Returns percentage of samples that satisifies the condition described in Corollary 4"""
        reduced_confs = get_reduced_confidences(confidences)
        reduced_predictions = self.calibration_method.get_calibrated_confidences(
            reduced_confs
        )
        reduced_predictions = reduced_predictions[:, 0]  # take only 0-class prediction
        n_classes = confidences.shape[1]

        return np.mean(
            reduced_predictions >= (1 / n_classes),
        )


class WeightedConfidenceReducedCalibration(ConfidenceReducedCalibration):
    def get_calibrated_confidences(self, confidences: np.ndarray):
        reduced_confs = get_reduced_confidences(confidences)
        reduced_predictions = self.calibration_method.get_calibrated_confidences(
            reduced_confs
        )
        reduced_predictions = reduced_predictions[:, 0]  # take only 0-class prediction
        # Compute other class predictions as proportional to original fractions
        reduced_predictions_complement = 1 - reduced_predictions

        # Mask max class predictions
        mask = np.zeros(confidences.shape, dtype=bool)
        mask[np.arange(len(confidences)), confidences.argmax(axis=1)] = True
        masked_confidences = confidences.copy()
        masked_confidences[mask] = 0.0

        # Compute proportions and distribute remaining confidences accordingly
        proportions = masked_confidences / masked_confidences.sum(axis=1, keepdims=True)
        # Replace NaN values with 1 / (K - 1)
        # This happens when all other predictions are 0
        proportions[np.where(np.isnan(proportions))] = 1.0 / (proportions.shape[1] - 1)

        calibrated_confidences = proportions * reduced_predictions_complement[:, None]

        argmax_indices = np.expand_dims(confidences.argmax(axis=1), axis=1)
        np.put_along_axis(
            calibrated_confidences, argmax_indices, reduced_predictions[:, None], axis=1
        )
        assert np.all(
            np.isclose(calibrated_confidences.sum(1), 1.0)
        ), "Calibrated confidences should all sum up to 1.0"
        assert (
            calibrated_confidences.shape == confidences.shape
        ), "Calibrated confidences' shape should be the same as the uncalibrated ones"
        return calibrated_confidences

    def satisfies_condition_percentage(self, confidences: np.ndarray):
        """Returns percentage of samples that satisifies the condition described in Corollary 5"""
        n_classes = confidences.shape[1]
        reduced_confs = get_reduced_confidences(confidences)
        reduced_predictions = self.calibration_method.get_calibrated_confidences(
            reduced_confs
        )
        reduced_predictions = reduced_predictions[:, 0]  # take only 0-class prediction

        # Mask max class predictions
        mask = np.zeros(confidences.shape, dtype=bool)
        mask[np.arange(len(confidences)), confidences.argmax(axis=1)] = True
        masked_confidences = confidences.copy()
        masked_confidences[mask] = 0.0

        # Compute proportions and distribute remaining confidences accordingly
        denominator = masked_confidences.sum(axis=1, keepdims=True)
        denominator = np.repeat(denominator, n_classes, axis=1)
        denominator = denominator + masked_confidences
        proportions = masked_confidences / denominator
        # Replace NaN values with 1 / (K - 1)
        # This happens when all other predictions are 0
        proportions[np.where(np.isnan(proportions))] = 1.0 / (n_classes - 1)

        return np.mean(
            np.logical_and.reduce(
                reduced_predictions[:, np.newaxis] >= proportions, axis=1
            )
        )


# HACK
class HistogramBinning(HistogramBinning):
    def __init__(self, bins=20):
        super().__init__(bins)
        self.bins = bins
        self.netcal_model.bins = bins
