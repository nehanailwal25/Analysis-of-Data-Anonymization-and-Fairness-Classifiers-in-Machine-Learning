from sklearn import metrics
import numpy as np

def metric(index, x_test, y_test, y_test_predicted):
    
    
    print("Accuracy:",metrics.accuracy_score(y_test, y_test_predicted))
    print("Precision:",metrics.precision_score(y_test, y_test_predicted))
    print("Recall:",metrics.recall_score(y_test, y_test_predicted))
    print("F1 score:",metrics.f1_score(y_test, y_test_predicted))
    
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
    TPR_1 = TP/(TP+FN)
    TNR_1 = TN/(FP+TN)
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
    
    TPR_0 = TP/(TP+FN)
    TNR_0 = TN/(FP+TN)
    print("DI: ", di(index, x_test, y_test, y_test_predicted))
    print("TPRB: ", TPR_1-TPR_0)
    print("TNRB: ", TNR_1-TNR_0)
    

    
def di(index, x_test, y_test, y_pred):
    
    a,b,c,d = 0, 0, 0, 0
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
def cd(index, x_test, y_test, clf):
    
    conf_z = 2.58
    #print(x_test.features)
    for i, val in enumerate(x_test.features):
        val[index] = (val[index] + 1)%2
        x_test.features[i, index] = val[index]
        
    #print(x_test.features)
    
    y_pred = clf.predict(x_test)
    count = 0
    res = []
    for i, val in enumerate(y_pred.labels):

        if(val != y_test[i]):
            count = count + 1
        res.append(val)
        res.append(y_test[i])
            
    cd = (count/x_test.features.shape[0])
    err = conf_z * math.sqrt((cd * (1 - cd)) / x_test.features.shape[0])
    print("CD:", cd)
    
    return np.array(res)