# Metrics function
from collections import OrderedDict
from fParam.metrics import ClassificationMetric

def compute_metrics(dataset_true, dataset_pred, 
                    unprivileged_groups, privileged_groups,
                    disp = True):
    """ Compute the key metrics """
    classified_metric_pred = ClassificationMetric(dataset_true,
                                                 dataset_pred, 
                                                 unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)
    
    metrics = OrderedDict()
    metrics["Balanced accuracy"] = 0.5*(classified_metric_pred.true_positive_rate()+
                                             classified_metric_pred.true_negative_rate())
    metrics["Statistical parity difference"] = classified_metric_pred.statistical_parity_difference()
    metrics["Disparate impact"] = classified_metric_pred.disparate_impact()
    metrics["Average odds difference"] = classified_metric_pred.average_odds_difference()
    metrics["Equal opportunity difference"] = classified_metric_pred.equal_opportunity_difference()
    metrics["Theil index"] = classified_metric_pred.theil_index()
    
    
    if disp:
        for k in metrics:
            print("%s = %.4f" % (k, metrics[k]))
    metrics["conf_mat_priv"] = classified_metric_pred.binary_confusion_matrix(privileged=True)
    metrics["conf_mat_prot"] = classified_metric_pred.binary_confusion_matrix(privileged=False)
    
    return metrics