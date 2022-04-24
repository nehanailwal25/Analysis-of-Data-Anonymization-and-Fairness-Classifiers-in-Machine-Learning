from itertools import product

import numpy as np

from fParam.metrics import BinaryLabelDatasetMetric, utils
from fParam.datasets import BinaryLabelDataset


class ClassificationMetric(BinaryLabelDatasetMetric):
    """Class for computing metrics based on two BinaryLabelDatasets. """

    def __init__(self, dataset, classified_dataset,
                 unprivileged_groups=None, privileged_groups=None):

        if not isinstance(dataset, BinaryLabelDataset):
            raise TypeError("'dataset' should be a BinaryLabelDataset")

        # sets self.dataset, self.unprivileged_groups, self.privileged_groups
        super(ClassificationMetric, self).__init__(dataset,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)

        if isinstance(classified_dataset, BinaryLabelDataset):
            self.classified_dataset = classified_dataset
        else:
            raise TypeError("'classified_dataset' should be a "
                            "BinaryLabelDataset.")

        # Verify if everything except the predictions and metadata are the same
        # for the two datasets
        with self.dataset.temporarily_ignore('labels', 'scores'):
            if self.dataset != self.classified_dataset:
                raise ValueError("The two datasets are expected to differ only "
                                 "in 'labels' or 'scores'.")

    def binary_confusion_matrix(self, privileged=None):
        """Compute the number of true/false positives/negatives, optionally
        conditioned on protected attributes. """
        condition = self._to_condition(privileged)

        return utils.compute_num_TF_PN(self.dataset.protected_attributes,
            self.dataset.labels, self.classified_dataset.labels,
            self.dataset.instance_weights,
            self.dataset.protected_attribute_names,
            self.dataset.favorable_label, self.dataset.unfavorable_label,
            condition=condition)

    def generalized_binary_confusion_matrix(self, privileged=None):
        """Compute the number of generalized true/false positives/negatives,
        optionally conditioned on protected attributes. Generalized counts are
        based on scores and not on the hard predictions. """
        condition = self._to_condition(privileged)

        return utils.compute_num_gen_TF_PN(self.dataset.protected_attributes,
            self.dataset.labels, self.classified_dataset.scores,
            self.dataset.instance_weights,
            self.dataset.protected_attribute_names,
            self.dataset.favorable_label, self.dataset.unfavorable_label,
            condition=condition)

    def num_true_positives(self, privileged=None):
        r"""Return the number of instances in the dataset where both the
        predicted and true labels are 'favorable' """
        return self.binary_confusion_matrix(privileged=privileged)['TP']

    def num_false_positives(self, privileged=None):
        return self.binary_confusion_matrix(privileged=privileged)['FP']

    def num_false_negatives(self, privileged=None):
        return self.binary_confusion_matrix(privileged=privileged)['FN']

    def num_true_negatives(self, privileged=None):
        return self.binary_confusion_matrix(privileged=privileged)['TN']

    def num_generalized_true_positives(self, privileged=None):
        """Return the generalized number of true positives. """
        return self.generalized_binary_confusion_matrix(
            privileged=privileged)['GTP']

    def num_generalized_false_positives(self, privileged=None):
        """Return the generalized number of false positives."""
        return self.generalized_binary_confusion_matrix(
            privileged=privileged)['GFP']

    def num_generalized_false_negatives(self, privileged=None):
        """Return the generalized number of false negatives. """
        return self.generalized_binary_confusion_matrix(
            privileged=privileged)['GFN']

    def num_generalized_true_negatives(self, privileged=None):
        """Return the generalized number of true negatives. """
        return self.generalized_binary_confusion_matrix(
            privileged=privileged)['GTN']

    def performance_measures(self, privileged=None):
        """Compute various performance measures on the dataset, optionally
        conditioned on protected attributes. """

        TP = self.num_true_positives(privileged=privileged)
        FP = self.num_false_positives(privileged=privileged)
        FN = self.num_false_negatives(privileged=privileged)
        TN = self.num_true_negatives(privileged=privileged)
        GTP = self.num_generalized_true_positives(privileged=privileged)
        GFP = self.num_generalized_false_positives(privileged=privileged)
        GFN = self.num_generalized_false_negatives(privileged=privileged)
        GTN = self.num_generalized_true_negatives(privileged=privileged)
        P = self.num_positives(privileged=privileged)
        N = self.num_negatives(privileged=privileged)

        return dict(
            TPR=TP / P, TNR=TN / N, FPR=FP / N, FNR=FN / P,
            GTPR=GTP / P, GTNR=GTN / N, GFPR=GFP / N, GFNR=GFN / P,
            PPV=TP / (TP+FP) if (TP+FP) > 0.0 else np.float64(0.0),
            NPV=TN / (TN+FN) if (TN+FN) > 0.0 else np.float64(0.0),
            FDR=FP / (FP+TP) if (FP+TP) > 0.0 else np.float64(0.0),
            FOR=FN / (FN+TN) if (FN+TN) > 0.0 else np.float64(0.0),
            ACC=(TP+TN) / (P+N) if (P+N) > 0.0 else np.float64(0.0)
        )

    def true_positive_rate(self, privileged=None):
        """Return the ratio of true positives to positive examples in the
        dataset. """
        return self.performance_measures(privileged=privileged)['TPR']

    def false_positive_rate(self, privileged=None):
        return self.performance_measures(privileged=privileged)['FPR']

    def false_negative_rate(self, privileged=None):
        return self.performance_measures(privileged=privileged)['FNR']

    def true_negative_rate(self, privileged=None):
        return self.performance_measures(privileged=privileged)['TNR']

    def generalized_true_positive_rate(self, privileged=None):
        """Return the ratio of generalized true positives to positive examples
        in the dataset. """
        return self.performance_measures(privileged=privileged)['GTPR']

    def generalized_false_positive_rate(self, privileged=None):
        return self.performance_measures(privileged=privileged)['GFPR']

    def generalized_false_negative_rate(self, privileged=None):
        return self.performance_measures(privileged=privileged)['GFNR']

    def generalized_true_negative_rate(self, privileged=None):
        return self.performance_measures(privileged=privileged)['GTNR']

    def positive_predictive_value(self, privileged=None):
        return self.performance_measures(privileged=privileged)['PPV']

    def false_discovery_rate(self, privileged=None):
        return self.performance_measures(privileged=privileged)['FDR']

    def false_omission_rate(self, privileged=None):
        return self.performance_measures(privileged=privileged)['FOR']

    def negative_predictive_value(self, privileged=None):
        return self.performance_measures(privileged=privileged)['NPV']

    def accuracy(self, privileged=None):
        return self.performance_measures(privileged=privileged)['ACC']

    def error_rate(self, privileged=None):
        return 1. - self.accuracy(privileged=privileged)

    def true_positive_rate_difference(self):
        return self.difference(self.true_positive_rate)

    def false_positive_rate_difference(self):
        return self.difference(self.false_positive_rate)

    def false_negative_rate_difference(self):
        return self.difference(self.false_negative_rate)

    def false_omission_rate_difference(self):
        return self.difference(self.false_omission_rate)

    def false_discovery_rate_difference(self):
        return self.difference(self.false_discovery_rate)

    def false_positive_rate_ratio(self):
        return self.ratio(self.false_positive_rate)

    def false_negative_rate_ratio(self):
        return self.ratio(self.false_negative_rate)

    def false_omission_rate_ratio(self):
        return self.ratio(self.false_omission_rate)

    def false_discovery_rate_ratio(self):
        return self.ratio(self.false_discovery_rate)

    def average_odds_difference(self):
        r"""Average of difference in FPR and TPR for unprivileged and privileged
        groups. """
        return 0.5 * (self.difference(self.false_positive_rate)
                    + self.difference(self.true_positive_rate))

    def average_abs_odds_difference(self):
        r"""Average of absolute difference in FPR and TPR for unprivileged and
        privileged groups. """
        return 0.5 * (np.abs(self.difference(self.false_positive_rate))
                    + np.abs(self.difference(self.true_positive_rate)))

    def error_rate_difference(self):
        r"""Difference in error rates for unprivileged and privileged groups. """
        return self.difference(self.error_rate)

    def error_rate_ratio(self):
        r"""Ratio of error rates for unprivileged and privileged groups. """
        return self.ratio(self.error_rate)

    def num_pred_positives(self, privileged=None):
        condition = self._to_condition(privileged)

        return utils.compute_num_pos_neg(
            self.classified_dataset.protected_attributes,
            self.classified_dataset.labels,
            self.classified_dataset.instance_weights,
            self.classified_dataset.protected_attribute_names,
            self.classified_dataset.favorable_label,
            condition=condition)

    def num_pred_negatives(self, privileged=None):
        condition = self._to_condition(privileged)

        return utils.compute_num_pos_neg(
            self.classified_dataset.protected_attributes,
            self.classified_dataset.labels,
            self.classified_dataset.instance_weights,
            self.classified_dataset.protected_attribute_names,
            self.classified_dataset.unfavorable_label,
            condition=condition)

    def selection_rate(self, privileged=None):
        return (self.num_pred_positives(privileged=privileged)
              / self.num_instances(privileged=privileged))

    def disparate_impact(self):
        return self.ratio(self.selection_rate)

    def statistical_parity_difference(self):
        return self.difference(self.selection_rate)

    def generalized_entropy_index(self, alpha=2):
        r"""Generalized entropy index is proposed as a unified individual and
        group fairness measure in [3]_.  With :math:`b_i = \hat{y}_i - y_i + 1`:

        .. math::

           \mathcal{E}(\alpha) = \begin{cases}
               \frac{1}{n \alpha (\alpha-1)}\sum_{i=1}^n\left[\left(\frac{b_i}{\mu}\right)^\alpha - 1\right],& \alpha \ne 0, 1,\\
               \frac{1}{n}\sum_{i=1}^n\frac{b_{i}}{\mu}\ln\frac{b_{i}}{\mu},& \alpha=1,\\
               -\frac{1}{n}\sum_{i=1}^n\ln\frac{b_{i}}{\mu},& \alpha=0.
           \end{cases}

        Args:
            alpha (int): Parameter that regulates the weight given to distances
                between values at different parts of the distribution.

        References:
            .. [3] T. Speicher, H. Heidari, N. Grgic-Hlaca, K. P. Gummadi, A. Singla, A. Weller, and M. B. Zafar,
               "A Unified Approach to Quantifying Algorithmic Unfairness: Measuring Individual and Group Unfairness via Inequality Indices,"
               ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2018.
        """
        y_pred = self.classified_dataset.labels.ravel()
        y_true = self.dataset.labels.ravel()
        y_pred = (y_pred == self.classified_dataset.favorable_label).astype(
            np.float64)
        y_true = (y_true == self.dataset.favorable_label).astype(np.float64)
        b = 1 + y_pred - y_true

        if alpha == 1:
            # moving the b inside the log allows for 0 values
            return np.mean(np.log((b / np.mean(b))**b) / np.mean(b))
        elif alpha == 0:
            return -np.mean(np.log(b / np.mean(b)) / np.mean(b))
        else:
            return np.mean((b / np.mean(b))**alpha - 1) / (alpha * (alpha - 1))

    def _between_group_generalized_entropy_index(self, groups, alpha=2):
        r"""Between-group generalized entropy index is proposed as a group
        fairness measure in [2]_ and is one of two terms that the generalized
        entropy index decomposes to.

        Args:
            groups (list): A list of groups over which to calculate this metric.
                Groups should be disjoint. By default, this will use the
                `privileged_groups` and `unprivileged_groups` as the only two
                groups.
            alpha (int): See :meth:`generalized_entropy_index`.

        References:
            .. [2] T. Speicher, H. Heidari, N. Grgic-Hlaca, K. P. Gummadi, A. Singla, A. Weller, and M. B. Zafar,
               "A Unified Approach to Quantifying Algorithmic Unfairness: Measuring Individual and Group Unfairness via Inequality Indices,"
               ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2018.
        """
        b = np.zeros(self.dataset.labels.size, dtype=np.float64)

        for group in groups:
            classified_group = utils.compute_boolean_conditioning_vector(
                self.classified_dataset.protected_attributes,
                self.classified_dataset.protected_attribute_names,
                condition=group)
            true_group = utils.compute_boolean_conditioning_vector(
                self.dataset.protected_attributes,
                self.dataset.protected_attribute_names,
                condition=group)
            # ignore if there are no members of this group present
            if not np.any(true_group):
                continue
            y_pred = self.classified_dataset.labels[classified_group].ravel()
            y_true = self.dataset.labels[true_group].ravel()
            y_pred = (y_pred == self.classified_dataset.favorable_label).astype(
                np.float64)
            y_true = (y_true == self.dataset.favorable_label).astype(np.float64)
            b[true_group] = np.mean(1 + y_pred - y_true)

        if alpha == 1:
            return np.mean(np.log((b / np.mean(b))**b) / np.mean(b))
        elif alpha == 0:
            return -np.mean(np.log(b / np.mean(b)) / np.mean(b))
        else:
            return np.mean((b / np.mean(b))**alpha - 1) / (alpha * (alpha - 1))

    def between_all_groups_generalized_entropy_index(self, alpha=2):
        """Between-group generalized entropy index that uses all combinations of
        groups based on `self.dataset.protected_attributes`. """
        all_values = list(map(np.concatenate, zip(
            self.dataset.privileged_protected_attributes,
            self.dataset.unprivileged_protected_attributes)))
        groups = [[dict(zip(self.dataset.protected_attribute_names, vals))]
                  for vals in product(*all_values)]
        return self._between_group_generalized_entropy_index(groups=groups,
            alpha=alpha)

    def between_group_generalized_entropy_index(self, alpha=2):
        """Between-group generalized entropy index that uses
        `self.privileged_groups` and `self.unprivileged_groups` as the only two
        groups. See :meth:`_between_group_generalized_entropy_index`.

        Args:
            alpha (int): See :meth:`generalized_entropy_index`.
        """
        groups = [self._to_condition(False), self._to_condition(True)]
        return self._between_group_generalized_entropy_index(groups=groups,
            alpha=alpha)

    def theil_index(self):
        r"""The Theil index is the :meth:`generalized_entropy_index` with
        :math:`\alpha = 1`.
        """
        return self.generalized_entropy_index(alpha=1)

    def coefficient_of_variation(self):
        r"""The coefficient of variation is two times the square root of the
        :meth:`generalized_entropy_index` with :math:`\alpha = 2`.
        """
        return 2 * np.sqrt(self.generalized_entropy_index(alpha=2))

    def between_group_theil_index(self):
        r"""The between-group Theil index is the
        :meth:`between_group_generalized_entropy_index` with :math:`\alpha = 1`.
        """
        return self.between_group_generalized_entropy_index(alpha=1)

    def between_group_coefficient_of_variation(self):
        r"""The between-group coefficient of variation is two times the square
        root of the :meth:`between_group_generalized_entropy_index` with
        :math:`\alpha = 2`.
        """
        return 2*np.sqrt(self.between_group_generalized_entropy_index(alpha=2))

    def between_all_groups_theil_index(self):
        r"""The between-group Theil index is the
        :meth:`between_all_groups_generalized_entropy_index` with
        :math:`\alpha = 1`.
        """
        return self.between_all_groups_generalized_entropy_index(alpha=1)

    def between_all_groups_coefficient_of_variation(self):
        r"""The between-group coefficient of variation is two times the square
        root of the :meth:`between_all_groups_generalized_entropy_index` with
        :math:`\alpha = 2`.
        """
        return 2 * np.sqrt(self.between_all_groups_generalized_entropy_index(
            alpha=2))

    def differential_fairness_bias_amplification(self, concentration=1.0):
        """Bias amplification is the difference in smoothed EDF between the
        classifier and the original dataset. Positive values mean the bias
        increased due to the classifier. """
        ssr = self._smoothed_base_rates(self.classified_dataset.labels,
                                        concentration)

        def pos_ratio(i, j):
            return abs(np.log(ssr[i]) - np.log(ssr[j]))

        def neg_ratio(i, j):
            return abs(np.log(1 - ssr[i]) - np.log(1 - ssr[j]))

        edf_clf = max(max(pos_ratio(i, j), neg_ratio(i, j))
                for i in range(len(ssr)) for j in range(len(ssr)) if i != j)
        edf_data = self.smoothed_empirical_differential_fairness(concentration)

        return edf_clf - edf_data

    # ============================== ALIASES ===================================
    def equal_opportunity_difference(self):
        """Alias of :meth:`true_positive_rate_difference`."""
        return self.true_positive_rate_difference()

    def power(self, privileged=None):
        """Alias of :meth:`num_true_positives`."""
        return self.num_true_positives(privileged=privileged)

    def precision(self, privileged=None):
        """Alias of :meth:`positive_predictive_value`."""
        return self.positive_predictive_value(privileged=privileged)

    def recall(self, privileged=None):
        """Alias of :meth:`true_positive_rate`."""
        return self.true_positive_rate(privileged=privileged)

    def sensitivity(self, privileged=None):
        """Alias of :meth:`true_positive_rate`."""
        return self.true_positive_rate(privileged=privileged)

    def specificity(self, privileged=None):
        """Alias of :meth:`true_negative_rate`."""
        return self.true_negative_rate(privileged=privileged)
