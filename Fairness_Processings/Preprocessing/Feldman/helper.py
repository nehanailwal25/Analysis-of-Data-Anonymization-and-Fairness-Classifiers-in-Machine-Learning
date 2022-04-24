import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
import time
warnings.filterwarnings('ignore')


def one_hot(df, col, pre):
    df_dummy = pd.get_dummies(df[col],prefix=pre,drop_first=True)
    df = pd.concat([df, df_dummy], axis=1)
    df = df.drop(col, axis=1) 
    
    return df

def metric(index, x_test, y_test, y_test_predicted):
    TP = 1
    FP = 0
    TN = 0
    FN = 0
    
    for i, val in enumerate(x_test):
        if(val[index] == 1):
            if y_test[i]==y_test_predicted[i]==1:
                TP += 1
            if y_test_predicted[i]==1 and y_test[i]!=y_test_predicted[i]:
                FP += 1
            if y_test[i]==y_test_predicted[i]== 0:
                TN += 1
            if y_test_predicted[i]==0 and y_test[i]!=y_test_predicted[i]:
                FN += 1
    TPR_0 = TP/(TP+FN)
    TNR_0 = TN/(FP+TN)
    
    #print(TP, FP, TN, FN)
    
    TP = 1
    FP = 0
    TN = 0
    FN = 0
    
    for i, val in enumerate(x_test):
        if(val[index] == 0):
            if y_test[i]==y_test_predicted[i]==1:
                TP += 1
            if y_test_predicted[i]==1 and y_test[i]!=y_test_predicted[i]:
                FP += 1
            if y_test[i]==y_test_predicted[i]==0:
                TN += 1
            if y_test_predicted[i]==0 and y_test[i]!=y_test_predicted[i]:
                FN += 1
    
    TPR = TP/(TP+FN)
    #print(TP, FP, TN, FN)
    TNR = TN/(FP+TN)
    print("Accuracy:",metrics.accuracy_score(y_test, y_test_predicted))
    print("Precision:",metrics.precision_score(y_test, y_test_predicted))
    print("Recall:",metrics.recall_score(y_test, y_test_predicted))
    print("F1:",metrics.f1_score(y_test, y_test_predicted))
    print("DI: ", di(index, x_test, y_test, y_test_predicted))
    print("TPRB:", TPR_0-TPR)
    print("TNRB:", TNR_0-TNR)

    
def di(index, x_test, y_test, y_pred):
    
    a,b,c,d = 1.0, 1, 0, 0
    for i, val in enumerate(x_test):
        if(val[index] == 0):
            if(y_pred[i] == 1):
                a += 1
            else:
                c += 1
        elif(val[index] == 1):
            if(y_pred[i] == 1):
                b += 1
            else:
                d += 1
    score = (a / (a + c)) / (b / (b + d))
    return score 

import math
def cd(index, x_test, clf):
    
    conf_z = 2.58
    x_test_new = np.zeros(shape=(x_test.shape[0]*2,x_test.shape[1]))
    
    for i, val in enumerate(x_test):
        x_test_new[i*2] = val
        val[index] = (val[index] + 1)%2
        x_test_new[i*2 +1] = val
    
    y_pred = clf.predict(np.delete(x_test_new, index, 1))
    count = 0
    for i, val in enumerate(y_pred):
        #print(val)
        if (i%2) == 1:
            continue
        if(val != y_pred[i+1]):
            count = count + 1
            
    cd = (count/x_test.shape[0])
    err = conf_z * math.sqrt((cd * (1 - cd)) / x_test.shape[0])
    print("CD:", cd)
    
    return y_pred

def adult_preprocess(df):
    def income(x):
        if x in ['<=50K', '0', 0]:
            return 0.0
        else:
            return 1.0
        
    def sex(x):
        if x in ['Male', "1", 1]:
            return 1.0
        else:
            return 0.0
        
        
    df['sex'] = df['sex'].apply(lambda x: sex(x))
    df['income'] = df['income'].apply(lambda x: income(x))
    return df


def Adult(f, f1=''):
    X_int = ['age', 'edu_level', 'hours_per_week']
    X_cat = [ 'marital_status', 'occupation','workclass', 'relationship', 'race', 'native_country']
    keep = [ 'edu_level', 'sex', 'income']
    S = ['sex']
    Y = ['income']
    
    df = pd.read_csv(f)
    df = df[keep]
    test = pd.read_csv("results_Feldman/adult_test_notrepaired.csv")
    test = test[keep]
    df = pd.concat([df, test])
    df = adult_preprocess(df)
    #df = df.dropna(how='any', axis=0) 
    for i in X_cat:
        if i in keep:
            df = one_hot(df, i, i) 
    
    X_train, X_test = train_test_split(df, test_size=0.3, shuffle=False, random_state=42)
    train_y = np.array(X_train['income'])
    train_s = np.array(X_train['sex'])
    X_train = X_train.drop(['income', 'sex'], axis=1)
    test_y = np.array(X_test['income'])
    test_s = np.array(X_test['sex'])
    X_test = X_test.drop(['income', 'sex'], axis=1)
    
    clf = LogisticRegression()
    clf.fit(X_train, train_y)
    y_pred = clf.predict(X_test)
    
    X_test['sex'] = test_s
    index = X_test.columns.get_loc('sex')
    metric(index, np.array(X_test), test_y, y_pred)
    y_cd = cd(index, np.array(X_test), clf)
    
    test = pd.read_csv("results_Feldman/adult_test_notrepaired.csv")
    test['pred'] = y_pred
    test.to_csv("results_Feldman/adult_test_repaired"+f1+".csv", index=False)
    np.savetxt("results_Feldman/adult_test_repaired_cd.csv", y_cd, delimiter=",")
    
    
def compute_metrics(dataset, f=None):
    if dataset == 'adult':
        Adult(f)
    elif dataset == 'compas':
        Compas(f)
    elif dataset == 'german':
        German(f)
    elif dataset == 'credit':
        Credit(f)
        
#compute_metrics('german', "results_Feldman/german_train_repaired.csv")