import numpy as np

from fParam.datasets import StructuredDataset

"""Base class for all structured datasets with binary labels."""
class BinaryLabelDataset(StructuredDataset):

    def __init__(self, favorable_label=1., unfavorable_label=0., **kwargs):

        self.favorable_label = float(favorable_label)
        self.unfavorable_label = float(unfavorable_label)

        super(BinaryLabelDataset, self).__init__(**kwargs)

    #Error checking and type validation
    def validate_dataset(self):
        # fix scores before validating
        if np.all(self.scores == self.labels):
            self.scores = np.float64(self.scores == self.favorable_label)

        super(BinaryLabelDataset, self).validate_dataset()

        # =========================== SHAPE CHECKING ===========================
        # Verify if the labels are only 1 column
        if self.labels.shape[1] != 1:
            raise ValueError("BinaryLabelDataset only supports single-column "
                "labels:\n\tlabels.shape = {}".format(self.labels.shape))

        # =========================== VALUE CHECKING ===========================
        # Check if the favorable and unfavorable labels match those in the dataset
        if (not set(self.labels.ravel()) <=
                set([self.favorable_label, self.unfavorable_label])):
            raise ValueError("The favorable and unfavorable labels provided do "
                             "not match the labels in the dataset.")
