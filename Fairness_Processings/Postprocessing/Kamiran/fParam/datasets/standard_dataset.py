from logging import warning

import numpy as np
import pandas as pd

from fParam.datasets import BinaryLabelDataset


class StandardDataset(BinaryLabelDataset):

    def __init__(self, df, label_name, favorable_classes,
                 protected_attribute_names, privileged_classes,
                 instance_weights_name='', scores_name='',
                 categorical_features=[], features_to_keep=[],
                 features_to_drop=[], na_values=[], custom_preprocessing=None,
                 metadata=None):

        # 2. Perform dataset-specific preprocessing
        if custom_preprocessing:
            df = custom_preprocessing(df)

        # 3. Drop unrequested columns
        features_to_keep = features_to_keep or df.columns.tolist()
        keep = (set(features_to_keep) | set(protected_attribute_names)
              | set(categorical_features) | set([label_name]))
        if instance_weights_name:
            keep |= set([instance_weights_name])
        df = df[sorted(keep - set(features_to_drop), key=df.columns.get_loc)]
        categorical_features = sorted(set(categorical_features) - set(features_to_drop), key=df.columns.get_loc)

        # 4. Remove any rows that have missing data.
        dropped = df.dropna()
        count = df.shape[0] - dropped.shape[0]
        if count > 0:
            print()
            #warning("Missing Data: {} rows removed from {}.".format(count,
            #        type(self).__name__))
        df = dropped

        # 5. Create a one-hot encoding of the categorical variables.
        df = pd.get_dummies(df, columns=categorical_features, prefix_sep='=')

        # 6. Map protected attributes to privileged/unprivileged
        privileged_protected_attributes = []
        unprivileged_protected_attributes = []
        for attr, vals in zip(protected_attribute_names, privileged_classes):
            privileged_values = [1.]
            unprivileged_values = [0.]
            if callable(vals):
                df[attr] = df[attr].apply(vals)
            elif np.issubdtype(df[attr].dtype, np.number):
                # this attribute is numeric; no remapping needed
                privileged_values = vals
                unprivileged_values = list(set(df[attr]).difference(vals))
            else:
                # find all instances which match any of the attribute values
                priv = np.logical_or.reduce(np.equal.outer(vals, df[attr].to_numpy()))
                df.loc[priv, attr] = privileged_values[0]
                df.loc[~priv, attr] = unprivileged_values[0]

            privileged_protected_attributes.append(
                np.array(privileged_values, dtype=np.float64))
            unprivileged_protected_attributes.append(
                np.array(unprivileged_values, dtype=np.float64))

        # 7. Make labels binary
        favorable_label = 1.
        unfavorable_label = 0.
        if callable(favorable_classes):
            df[label_name] = df[label_name].apply(favorable_classes)
        elif np.issubdtype(df[label_name], np.number) and len(set(df[label_name])) == 2:
            # labels are already binary; don't change them
            favorable_label = favorable_classes[0]
            unfavorable_label = set(df[label_name]).difference(favorable_classes).pop()
        else:
            # find all instances which match any of the favorable classes
            pos = np.logical_or.reduce(np.equal.outer(favorable_classes, 
                                                      df[label_name].to_numpy()))
            df.loc[pos, label_name] = favorable_label
            df.loc[~pos, label_name] = unfavorable_label

        super(StandardDataset, self).__init__(df=df, label_names=[label_name],
            protected_attribute_names=protected_attribute_names,
            privileged_protected_attributes=privileged_protected_attributes,
            unprivileged_protected_attributes=unprivileged_protected_attributes,
            instance_weights_name=instance_weights_name,
            scores_names=[scores_name] if scores_name else [],
            favorable_label=favorable_label,
            unfavorable_label=unfavorable_label, metadata=metadata)
