# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 21:16:58 2018

@author: Yhi
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn import neighbors
from sklearn import tree
from sklearn import ensemble
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
x=PTrain_ad.iloc[:,1:]
y=PTrain_ad.iloc[:,1]
x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.25,random_state=0)

#%% Brenchmark model

para_lo=[{'penalty':['l1','l2'],
                'C':np.logspace(-1,1,10),
                'solver':['liblinear'],
                'multi_class':['ovr']},
                {'penalty':['l2'],
                 'C':np.logspace(-1,1,20),
                'solver':['lbfgs'],
                'multi_class':['ovr','multinomial']}]

logcv=GridSearchCV(LogisticRegression(),para_lo,cv=10,scoring='roc_auc')
log=logcv.fit(x_train,y_train)
yyy=log.predict(x_val)
log.coef_
print("Number of defaults in test set: {0}".format(sum(y_val)))
print("Number of defaults in train set: {0}".format(sum(y_pred)))
print(accuracy_score(y_test,yyy))
print(confusion_matrix(y_test,yyy))
print(classification_report(y_test,yyy,digits=3))
print(clf.best_estimator_)
#%% RandomForest
para = [{'n_estimators':[110,120],
'':['entropy','gini'],
#'max_depth':[12,18,24],
'min_samples_split':[40], 
#'min_weight_fraction_leaf':[0.1,0.3,0.5],
'max_features':[4]
}]     
clf=GridSearchCV(RandomForestClassifier(
        bootstrap=True, class_weight=None, max_leaf_nodes=None, criterion='entropy',
        max_depth=12,min_samples_split=2,min_samples_leaf=2,
         min_weight_fraction_leaf=0.0, 
        n_jobs=1, oob_score=True, random_state=0, verbose=0,
        warm_start=False),para, scoring='roc_auc',cv=10)
clf=clf.fit(x_trains,y_train)  
yyy=clf.predict(x_tests)
print(accuracy_score(y_test,yyy))
print(confusion_matrix(y_test,yyy))
print(classification_report(y_test,yyy,digits=3))
print(clf.best_estimator_)