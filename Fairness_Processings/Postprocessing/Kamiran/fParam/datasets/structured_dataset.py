from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from logging import warning

import numpy as np
import pandas as pd

from fParam.datasets import Dataset

#Base class for all structured datasets
class StructuredDataset(Dataset):

    def __init__(self, df, label_names, protected_attribute_names,
                 instance_weights_name=None, scores_names=[],
                 unprivileged_protected_attributes=[],
                 privileged_protected_attributes=[], metadata=None):
        if df is None:
            raise TypeError("Must provide a pandas DataFrame representing "
                            "the data (features, labels, protected attributes)")
        if df.isna().any().any():
            raise ValueError("Input DataFrames cannot contain NA values.")
        try:
            df = df.astype(np.float64)
        except ValueError as e:
            print("ValueError: {}".format(e))
            raise ValueError("DataFrame values must be numerical.")

        # Convert all column names to strings
        df.columns = df.columns.astype(str).tolist()
        label_names = list(map(str, label_names))
        protected_attribute_names = list(map(str, protected_attribute_names))

        self.feature_names = [n for n in df.columns if n not in label_names
                              and (not scores_names or n not in scores_names)
                              and n != instance_weights_name]
        self.label_names = label_names
        self.features = df[self.feature_names].values.copy()
        self.labels = df[self.label_names].values.copy()
        self.instance_names = df.index.astype(str).tolist()

        if scores_names:
            self.scores = df[scores_names].values.copy()
        else:
            self.scores = self.labels.copy()

        df_prot = df.loc[:, protected_attribute_names]
        self.protected_attribute_names = df_prot.columns.astype(str).tolist()
        self.protected_attributes = df_prot.values.copy()

        # Infer the privileged and unprivileged values in not provided
        if unprivileged_protected_attributes and privileged_protected_attributes:
            self.unprivileged_protected_attributes = unprivileged_protected_attributes
            self.privileged_protected_attributes = privileged_protected_attributes
        else:
            self.unprivileged_protected_attributes = [
                np.sort(np.unique(df_prot[attr].values))[:-1]
                for attr in self.protected_attribute_names]
            self.privileged_protected_attributes = [
                np.sort(np.unique(df_prot[attr].values))[-1:]
                for attr in self.protected_attribute_names]

        if instance_weights_name:
            self.instance_weights = df[instance_weights_name].values.copy()
        else:
            self.instance_weights = np.ones_like(self.instance_names,
                                                 dtype=np.float64)

        # always ignore metadata and ignore_fields
        self.ignore_fields = {'metadata', 'ignore_fields'}

        # sets metadata
        super(StructuredDataset, self).__init__(df=df, label_names=label_names,
                                                protected_attribute_names=protected_attribute_names,
                                                instance_weights_name=instance_weights_name,
                                                unprivileged_protected_attributes=unprivileged_protected_attributes,
                                                privileged_protected_attributes=privileged_protected_attributes,
                                                metadata=metadata)

    #subset of dataset based on indexes
    def subset(self, indexes):
        # convert each element of indexes to string
        indexes_str = [self.instance_names[i] for i in indexes]
        subset = self.copy()
        subset.instance_names = indexes_str
        subset.features = self.features[indexes]
        subset.labels = self.labels[indexes]
        subset.instance_weights = self.instance_weights[indexes]
        subset.protected_attributes = self.protected_attributes[indexes]
        subset.scores = self.scores[indexes]
        return subset

    #Equality comparison for StructuredDatasets
    def __eq__(self, other):
        if not isinstance(other, StructuredDataset):
            return False

        def _eq(x, y):
            if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
                return np.all(x == y)
            elif isinstance(x, list) and isinstance(y, list):
                return len(x) == len(y) and all(_eq(xi, yi) for xi, yi in zip(x, y))
            return x == y

        return all(_eq(self.__dict__[k], other.__dict__[k])
                   for k in self.__dict__.keys() if k not in self.ignore_fields)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        # return repr(self.metadata)
        return str(self)

    def __str__(self):
        df, _ = self.convert_to_dataframe()
        df.insert(0, 'instance_weights', self.instance_weights)
        highest_level = ['instance weights'] + \
                        ['features'] * len(self.feature_names) + \
                        ['labels'] * len(self.label_names)
        middle_level = [''] + \
                       ['protected attribute'
                        if f in self.protected_attribute_names else ''
                        for f in self.feature_names] + \
                       [''] * len(self.label_names)
        lowest_level = [''] + self.feature_names + [''] * len(self.label_names)
        df.columns = pd.MultiIndex.from_arrays(
            [highest_level, middle_level, lowest_level])
        df.index.name = 'instance names'
        return str(df)

    #Error checking and type validation
    def validate_dataset(self):
        super(StructuredDataset, self).validate_dataset()

        # =========================== TYPE CHECKING ============================
        for f in [self.features, self.protected_attributes, self.labels,
                  self.scores, self.instance_weights]:
            if not isinstance(f, np.ndarray):
                raise TypeError("'{}' must be an np.ndarray.".format(f.__name__))

        # convert ndarrays to float64
        self.features = self.features.astype(np.float64)
        self.protected_attributes = self.protected_attributes.astype(np.float64)
        self.labels = self.labels.astype(np.float64)
        self.instance_weights = self.instance_weights.astype(np.float64)

        # =========================== SHAPE CHECKING ===========================
        if len(self.labels.shape) == 1:
            self.labels = self.labels.reshape((-1, 1))
        try:
            self.scores.reshape(self.labels.shape)
        except ValueError as e:
            print("ValueError: {}".format(e))
            raise ValueError("'scores' should have the same shape as 'labels'.")
        if not self.labels.shape[0] == self.features.shape[0]:
            raise ValueError("Number of labels must match number of instances:"
                             "\n\tlabels.shape = {}\n\tfeatures.shape = {}".format(
                self.labels.shape, self.features.shape))
        if not self.instance_weights.shape[0] == self.features.shape[0]:
            raise ValueError("Number of weights must match number of instances:"
                             "\n\tinstance_weights.shape = {}\n\tfeatures.shape = {}".format(
                self.instance_weights.shape, self.features.shape))

        # =========================== VALUE CHECKING ===========================
        if np.any(np.logical_or(self.scores < 0., self.scores > 1.)):
            warning("'scores' has no well-defined meaning out of range [0, 1].")

        for i in range(len(self.privileged_protected_attributes)):
            priv = set(self.privileged_protected_attributes[i])
            unpriv = set(self.unprivileged_protected_attributes[i])
            # check for duplicates
            if priv & unpriv:
                raise ValueError("'privileged_protected_attributes' and "
                                 "'unprivileged_protected_attributes' should not share any "
                                 "common elements:\n\tBoth contain {} for feature {}".format(
                    list(priv & unpriv), self.protected_attribute_names[i]))
            # check for unclassified values
            if not set(self.protected_attributes[:, i]) <= (priv | unpriv):
                raise ValueError("All observed values for protected attributes "
                                 "should be designated as either privileged or unprivileged:"
                                 "\n\t{} not designated for feature {}".format(
                    list(set(self.protected_attributes[:, i])
                         - (priv | unpriv)),
                    self.protected_attribute_names[i]))
            # warn for unobserved values
            if not (priv | unpriv) <= set(self.protected_attributes[:, i]):
                warning("{} listed but not observed for feature {}".format(
                    list((priv | unpriv) - set(self.protected_attributes[:, i])),
                    self.protected_attribute_names[i]))

    @contextmanager
    def temporarily_ignore(self, *fields):

        old_ignore = deepcopy(self.ignore_fields)
        self.ignore_fields |= set(fields)
        try:
            yield
        finally:
            self.ignore_fields = old_ignore

    #StructuredDataset: New aligned dataset
    def align_datasets(self, other):

        if (set(self.feature_names) != set(other.feature_names) or
                set(self.label_names) != set(other.label_names) or
                set(self.protected_attribute_names)
                != set(other.protected_attribute_names)):
            raise ValueError(
                "feature_names, label_names, and protected_attribute_names "
                "should match between this and other dataset.")

        # New dataset
        new = other.copy()

        # re-order the columns of the new dataset
        feat_inds = [new.feature_names.index(f) for f in self.feature_names]
        label_inds = [new.label_names.index(f) for f in self.label_names]
        prot_inds = [new.protected_attribute_names.index(f)
                     for f in self.protected_attribute_names]

        new.features = new.features[:, feat_inds]
        new.labels = new.labels[:, label_inds]
        new.scores = new.scores[:, label_inds]
        new.protected_attributes = new.protected_attributes[:, prot_inds]

        new.privileged_protected_attributes = [
            new.privileged_protected_attributes[i] for i in prot_inds]
        new.unprivileged_protected_attributes = [
            new.unprivileged_protected_attributes[i] for i in prot_inds]
        new.feature_names = deepcopy(self.feature_names)
        new.label_names = deepcopy(self.label_names)
        new.protected_attribute_names = deepcopy(self.protected_attribute_names)

        return new

    # TODO: Should we store the protected attributes as a separate dataframe
    def convert_to_dataframe(self, de_dummy_code=False, sep='=',
                             set_category=True):

        df = pd.DataFrame(np.hstack((self.features, self.labels)),
                          columns=self.feature_names + self.label_names,
                          index=self.instance_names)
        df.loc[:, self.protected_attribute_names] = self.protected_attributes

        # De-dummy code if necessary
        if de_dummy_code:
            df = self._de_dummy_code_df(df, sep=sep, set_category=set_category)
            if 'label_maps' in self.metadata:
                for i, label in enumerate(self.label_names):
                    df[label] = df[label].replace(self.metadata['label_maps'][i])
            if 'protected_attribute_maps' in self.metadata:
                for i, prot_attr in enumerate(self.protected_attribute_names):
                    df[prot_attr] = df[prot_attr].replace(
                        self.metadata['protected_attribute_maps'][i])

        # Attributes
        attributes = {
            "feature_names": self.feature_names,
            "label_names": self.label_names,
            "protected_attribute_names": self.protected_attribute_names,
            "instance_names": self.instance_names,
            "instance_weights": self.instance_weights,
            "privileged_protected_attributes": self.privileged_protected_attributes,
            "unprivileged_protected_attributes": self.unprivileged_protected_attributes
        }

        return df, attributes


    #Exporting the dataset and supporting attributes
    def export_dataset(self, export_metadata=False):

        if export_metadata:
            raise NotImplementedError("The option to export metadata has not been implemented yet")

        return None

    #Import the dataset and supporting attributes
    def import_dataset(self, import_metadata=False):
        if import_metadata:
            raise NotImplementedError("The option to import metadata has not been implemented yet")
        return None

    #Splitting this dataset into multiple partitions
    def split(self, num_or_size_splits, shuffle=False, seed=None):

        # Set seed
        if seed is not None:
            np.random.seed(seed)

        n = self.features.shape[0]
        if isinstance(num_or_size_splits, list):
            num_folds = len(num_or_size_splits) + 1
            if num_folds > 1 and all(x <= 1. for x in num_or_size_splits):
                num_or_size_splits = [int(x * n) for x in num_or_size_splits]
        else:
            num_folds = num_or_size_splits

        order = list(np.random.permutation(n) if shuffle else range(n))
        folds = [self.copy() for _ in range(num_folds)]

        features = np.array_split(self.features[order], num_or_size_splits)
        labels = np.array_split(self.labels[order], num_or_size_splits)
        scores = np.array_split(self.scores[order], num_or_size_splits)
        protected_attributes = np.array_split(self.protected_attributes[order],
                                              num_or_size_splits)
        instance_weights = np.array_split(self.instance_weights[order],
                                          num_or_size_splits)
        instance_names = np.array_split(np.array(self.instance_names)[order],
                                        num_or_size_splits)
        for fold, feats, labs, scors, prot_attrs, inst_wgts, inst_name in zip(
                folds, features, labels, scores, protected_attributes, instance_weights,
                instance_names):
            fold.features = feats
            fold.labels = labs
            fold.scores = scors
            fold.protected_attributes = prot_attrs
            fold.instance_weights = inst_wgts
            fold.instance_names = list(map(str, inst_name))
            fold.metadata = fold.metadata.copy()
            fold.metadata.update({
                'transformer': '{}.split'.format(type(self).__name__),
                'params': {'num_or_size_splits': num_or_size_splits,
                           'shuffle': shuffle},
                'previous': [self]
            })

        return folds

    @staticmethod
    def _de_dummy_code_df(df, sep="=", set_category=False):

        feature_names_dum_d, feature_names_nodum = \
            StructuredDataset._parse_feature_names(df.columns)
        df_new = pd.DataFrame(index=df.index, columns=feature_names_nodum + list(feature_names_dum_d.keys()))

        for fname in feature_names_nodum:
            df_new[fname] = df[fname].values.copy()

        for fname, vl in feature_names_dum_d.items():
            for v in vl:
                df_new.loc[df[fname + sep + str(v)] == 1, fname] = str(v)

        if set_category:
            for fname in feature_names_dum_d.keys():
                df_new[fname] = df_new[fname].astype('category')

        return df_new

    @staticmethod
    #Parsing feature names to ordinary and dummy coded candidates
    def _parse_feature_names(feature_names, sep="="):

        feature_names_dum_d = defaultdict(list)
        feature_names_nodum = list()
        for fname in feature_names:
            if sep in fname:
                fname_dum, v = fname.split(sep, 1)
                feature_names_dum_d[fname_dum].append(v)
            else:
                feature_names_nodum.append(fname)

        return feature_names_dum_d, feature_names_nodum
