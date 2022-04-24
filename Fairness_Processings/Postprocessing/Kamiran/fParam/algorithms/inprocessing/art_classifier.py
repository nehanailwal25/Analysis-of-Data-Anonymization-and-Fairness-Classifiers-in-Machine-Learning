import numpy as np

from fParam.datasets import BinaryLabelDataset
from fParam.algorithms import Transformer


class ARTClassifier(Transformer):

    def __init__(self, art_classifier):
        """Initialize ARTClassifier."""
        super(ARTClassifier, self).__init__(art_classifier=art_classifier)
        self._art_classifier = art_classifier

    def fit(self, dataset, batch_size=128, nb_epochs=20):
        """Train a classifer on the input."""
        self._art_classifier.fit(dataset.features, dataset.labels,
                                 batch_size=batch_size, nb_epochs=nb_epochs)
        return self

    def predict(self, dataset, logits=False):
        """Perform prediction for the input."""
        pred_labels = self._art_classifier.predict(dataset.features,
            dataset.labels, logits=logits)

        if isinstance(dataset, BinaryLabelDataset):
            pred_labels = np.argmax(pred_labels, axis=1).reshape((-1, 1))

        pred_dataset = dataset.copy()
        pred_dataset.labels = pred_labels

        return pred_dataset
