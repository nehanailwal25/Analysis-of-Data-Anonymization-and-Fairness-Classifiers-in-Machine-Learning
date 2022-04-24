import os,sys
import utils as ut
import numpy as np
import pandas as pd
from random import seed, shuffle
import loss_funcs as lf # loss funcs that can be optimized subject to various constraints
import time

SEED = 1122334455
seed(SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)

def Zafar(dataset):
    if dataset == 'adult':
        test_adult_data()
    elif dataset == 'compas':
        test_compas_data()
    else:
        test_german_data()
    
    return


# Loading the adult dataset
def load_adult_data(load_data_size=None):


    attrs = ['age', 'workclass', 'edu_level', 'marital_status', 'occupation', 'relationship', 
             'race', 'sex', 'hours_per_week', 'native_country'] # all attributes 
    int_attrs = ['age', 'edu_level', 'hours_per_week'] # attributes with integer values -- the rest are categorical
    
    sensitive_attrs = ['sex'] # the fairness constraints will be used for this feature
    
    attrs_to_ignore = ['sex', 'race'] # sex and race are sensitive feature so we will not use them in classification, we will not consider fnlwght for classification since its computed externally and it highly predictive for the class (for details, see documentation of the adult data)
    
    attrs_for_classification = set(attrs) - set(attrs_to_ignore)

    data_files = ['dataset/Adult.csv']#["adult.data", "adult.test"]


    X = []
    y = []
    x_control = {}
    
    attrs_to_vals = {} # will store the values for each attribute for all users
    for k in attrs:
        if k in sensitive_attrs:
            x_control[k] = []
        elif k in attrs_to_ignore:
            pass
        else:
            attrs_to_vals[k] = []

    for f in data_files:
    
        for line in open(f):
            line = line.strip()
            if line.startswith("age") or line == "":
                continue
            line = line.split(",")
            if "?" in line: # if a line has missing attributes, ignore it
                continue
            
            class_label = line[-1]
            if class_label in ["<=50K.", "<=50K", "0"]:
                class_label = -1
            elif class_label in [">50K.", ">50K", "1"]:
                class_label = +1
            else:
                raise Exception("Invalid class label value")
            y.append(class_label)


            for i in range(0,len(line)-1):
                attr_name = attrs[i]
                attr_val = line[i]
                # reducing dimensionality of some very sparse features
                if attr_name == "native_country":
                    if attr_val!="United-States":
                        attr_val = "Non-United-Stated"
                elif attr_name == "education":
                    if attr_val in ["Preschool", "1st-4th", "5th-6th", "7th-8th"]:
                        attr_val = "prim-middle-school"
                    elif attr_val in ["9th", "10th", "11th", "12th"]:
                        attr_val = "high-school"

                if attr_name in sensitive_attrs:
                    x_control[attr_name].append(attr_val)
                elif attr_name in attrs_to_ignore:
                    pass
                else:
                    attrs_to_vals[attr_name].append(attr_val)

    def convert_attrs_to_ints(d): # discretize the string attributes
        for attr_name, attr_vals in d.items():
            if attr_name in int_attrs: continue
            uniq_vals = sorted(list(set(attr_vals))) # get unique values

            # compute integer codes for the unique values
            val_dict = {}
            for i in range(0,len(uniq_vals)):
                val_dict[uniq_vals[i]] = i

            # replace the values with their integer encoding
            for i in range(0,len(attr_vals)):
                attr_vals[i] = val_dict[attr_vals[i]]
            d[attr_name] = attr_vals

    
    # convert the discrete values to their integer representations
    convert_attrs_to_ints(x_control)
    convert_attrs_to_ints(attrs_to_vals)
    

    # if the integer vals are not binary, we need to get one-hot encoding for them
    for attr_name in attrs_for_classification:
        attr_vals = attrs_to_vals[attr_name]
        if attr_name in int_attrs or attr_name == "native_country" or attr_name == "sex": 
            # the way we encoded native country, its binary now so no need to apply one hot encoding on it
            X.append(attr_vals)

        else: 
            attr_vals, index_dict = ut.get_one_hot_encoding(attr_vals)
            for inner_col in attr_vals.T:                
                X.append(inner_col) 
                

    # convert to numpy arrays for easy handline
    X = np.array(X, dtype=float).T
    y = np.array(y, dtype = float)
    for k, v in x_control.items(): x_control[k] = np.array(v, dtype=float)
        
        
    # shuffle the data
    perm = list(range(0,len(y))) # shuffle the data before creating each fold
    #shuffle(perm)
    X = X[perm]
    y = y[perm]
    for k in x_control.keys():
        x_control[k] = x_control[k][perm]

    # see if we need to subsample the data
    if load_data_size is not None:
        print ("Loading only %d examples from the data" % load_data_size)
        X = X[:load_data_size]
        y = y[:load_data_size]
        for k in x_control.keys():
            x_control[k] = x_control[k][:load_data_size]

    return X, y, x_control
    


def test_adult_data():
	

	""" Load the adult data """
	X, y, x_control = load_adult_data(load_data_size=None) 
	ut.compute_p_rule(x_control["sex"], y) # compute the p-rule in the original data

	""" Split the data into train and test """
	X = ut.add_intercept(X) # add intercept to X before applying the linear classifier
	train_fold_size = 0.7
	split_point = int(round(float(X.shape[0]) * train_fold_size))
	x_train, y_train, x_control_train, x_test, y_test, x_control_test = \
    ut.split_into_train_test(X, y, x_control, train_fold_size)
    
	apply_fairness_constraints = None
	apply_accuracy_constraint = None
	sep_constraint = None

	loss_function = lf._logistic_loss
	sensitive_attrs = ["sex"]
	sensitive_attrs_to_cov_thresh = {}
	gamma = None

	def train_test_classifier():
		w = ut.train_model(x_train, y_train, x_control_train, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)
		train_score, test_score, correct_answers_train, correct_answers_test, y_pred =\
        ut.check_accuracy(w, x_train, y_train, x_test, y_test, x_control_test, None, None)
		distances_boundary_test = (np.dot(x_test, w)).tolist()
		all_class_labels_assigned_test = np.sign(distances_boundary_test)
		correlation_dict_test = ut.get_correlations(None, None, all_class_labels_assigned_test, x_control_test, sensitive_attrs)
		cov_dict_test = ut.print_covariance_sensitive_attrs(None, x_test, distances_boundary_test, x_control_test, sensitive_attrs)
		p_rule = ut.print_classifier_fairness_stats([test_score], [correlation_dict_test], [cov_dict_test], sensitive_attrs[0])	
		return w, p_rule, test_score, y_pred


	print ("== Unconstrained (original) classifier ==")
	# all constraint flags are set to 0 since we want to train an unconstrained (original) classifier
	apply_fairness_constraints = 0
	apply_accuracy_constraint = 0
	sep_constraint = 0
	w_uncons, p_uncons, acc_uncons, _ = train_test_classifier()
	
	print("==Optimizing fairness under accuracy==")
	apply_fairness_constraints = 1 
	apply_accuracy_constraint = 0
	sep_constraint = 0
	sensitive_attrs_to_cov_thresh = {"sex":0}
	print()
	start = time.time()
	w_f_cons, p_f_cons, acc_f_cons, y_pred  = train_test_classifier()
	end = time.time()
	print("Time: ", end - start)
	df = pd.read_csv("dataset/Adult.csv")
	df = df.iloc[split_point:, :]
	y_pred[y_pred < 0] = 0
	df['pred'] = y_pred
	df.to_csv("results_zafarDI/adult_test_repaired_accuracy.csv", index=False)

	

	print("==Optimizing accuracy under fairness==")
	apply_fairness_constraints = 0 
	apply_accuracy_constraint = 1 
	sep_constraint = 0
	gamma = 0.5 # gamma controls how much loss in accuracy 
	start = time.time()
	w_a_cons, p_a_cons, acc_a_cons, y_pred = train_test_classifier()	
	end = time.time()
	print("Time: ", end - start)
	df = pd.read_csv("dataset/Adult.csv")
	df = df.iloc[split_point:, :]
	y_pred[y_pred < 0] = 0
	df['pred'] = y_pred
	df.to_csv("results_zafarDI/adult_test_repaired_fairness.csv", index=False)

	return


    

   
