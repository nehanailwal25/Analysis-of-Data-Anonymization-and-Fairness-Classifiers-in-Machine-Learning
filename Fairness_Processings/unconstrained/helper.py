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
    TP = 0
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
    
    TP = 0
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
    TNR = TN/(FP+TN)
    print("Accuracy:",metrics.accuracy_score(y_test, y_test_predicted))
    print("Precision:",metrics.precision_score(y_test, y_test_predicted))
    print("Recall:",metrics.recall_score(y_test, y_test_predicted))
    print("F1:",metrics.f1_score(y_test, y_test_predicted))
    print("DI: ", di(index, x_test, y_test, y_test_predicted))
    print("TPRB:", TPR_0-TPR)
    print("TNRB:", TNR_0-TNR)
    

    
    
def di(index, x_test, y_test, y_pred):
    
    a,b,c,d = 0.0, 0, 0, 0
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
    
    y_pred = clf.predict(x_test_new)
    count = 0
    for i, val in enumerate(y_pred):
        #print(val)
        if (i%2) == 1:
            continue
        if(val != y_pred[i+1]):
            count = count + 1
            
    cd = (count/x_test.shape[0])
    err = conf_z * math.sqrt((cd * (1 - cd)) / x_test.shape[0])
    print("CD:", cd, "margin of error:", err)
    
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
        
    def country_bin(x):
        if (x == 'United-States'):
            return "United-States"
        else:
            return "Non-US"
        
        
    df['sex'] = df['sex'].apply(lambda x: sex(x))
    df['income'] = df['income'].apply(lambda x: income(x))
    df['native_country'] = df['native_country'].apply(lambda x: country_bin(x))
    return df


def Adult(f):
    X_int = ['age', 'edu_level', 'hours_per_week']
    X_cat = [ 'marital_status', 'occupation','workclass', 'relationship', 'race', 'native_country']
    S = ['sex']
    Y = ['income']
    keep = X_int + X_cat + S + Y
    
    df = pd.read_csv(f)
    df = df[keep]
    test = pd.read_csv("data/adult_test.csv")
    test = test[keep]
    df = pd.concat([df, test])
    df = adult_preprocess(df)
    #df = df.dropna(how='any', axis=0) 
    for i in X_cat:
        if i in keep:
            df = one_hot(df, i, i) 
    
    X_train, X_test = train_test_split(df, test_size=0.3, shuffle=False)
    train_y = np.array(X_train['income'])
    X_train = X_train.drop(['income'], axis=1)
    test_y = np.array(X_test['income'])
    X_test = X_test.drop(['income'], axis=1)
    
    index = X_train.columns.get_loc('sex')
    clf = LogisticRegression(solver="liblinear")
    clf.fit(X_train, train_y)
    y_pred = clf.predict(X_test)
    metric(index, np.array(X_test), test_y, y_pred)
    y_cd = cd(index,  np.array(X_test), clf)
    
    test = pd.read_csv("data/adult_test.csv")
    test['pred'] = y_pred
    test.to_csv("results_unconstrained/adult_test_repaired.csv", index=False)
    np.savetxt("results_unconstrained/adult_test_repaired_cd.csv", y_cd, delimiter=",")


    
def compute_metrics(dataset, f=None):
    if dataset == 'adult':
        Adult(f)
        

#compute_metrics('compas', "results_Kamiran/compas_train_repaired.csv")

    