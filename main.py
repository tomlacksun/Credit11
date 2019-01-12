# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 13:51:46 2018



@author: Yhi

We use xgboost for this assignment, please install the xgboost library
the install code by pip is : pip3 install xgboost
Sorry for inconvenience :)
To run the code more quickly, i invalidate all the code for gridsearch CV for 
the final version, with the best estimator model from gridsearch CV before

"""
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.tree import DecisionTreeClassifier
if __name__=='__main__':
    #%% Import the data
    PTrain_o=pd.read_csv('ProjectTrain.csv')
    PTrainB=pd.read_csv('ProjectTrain_Bureau.csv').rename(columns={'AMT_ANNUITY':'AMT_ANNUITY_B'})
    PTest=pd.read_csv('ProjectTest.csv')
    PTestB=pd.read_csv('ProjectTest_Bureau.csv').rename(columns={'AMT_ANNUITY':'AMT_ANNUITY_B'})
    #%%
    #import relevant packages
    np.set_printoptions(threshold=np.inf)
    #read csv data
    data=PTrain_o
    #check data types
    data.dtypes
    data.get_dtype_counts()
    #check missing vlaue
    data.count()
    data.describe()
    #top missing words

    #%%
    #remove the missing feature
    PTrain=PTrain_o.iloc[:,1:]
    PTrain.head()
    PTrain.describe()
    corr=PTrain.corr().abs()
    corr_ad=corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
    to_drop = [column for column in corr_ad.columns if any(corr_ad[column] > 0.8)]
    print('%d columns with more than 0.8 correaltion is removed.' % (len(to_drop)))
    #renive the feature with high correlation
    PTrain = PTrain.drop(columns = to_drop)
    miss = PTrain.isnull().sum()
    missper = (PTrain.isnull().sum()/len(PTrain))
    missing_data = pd.DataFrame({'Total':miss,'Percent':missper}).reset_index().sort_values(by='Total',ascending=False)
    missing11=missing_data[missing_data['Percent']>0.5]['index']
    print('%d columns with more than 50%% missing values is removed.' % len(missing11))
    PTrain = PTrain.drop(columns = missing11)
    #define the clean function

    #%%

    #%%
    #Clean the original data with full features
    PTrain_ad=Deal(PTrain)
    #rename the columns with the same name
    x111=PTrain_ad.columns.duplicated(keep = False)
    x222=pd.DataFrame([PTrain_ad.columns,x111],index=['name','value']).T
    x333=x222[x222['value']==True]['name'].reset_index()
    xxxxx=list(PTrain_ad.columns)
    xxxxx[58]='Y_'
    xxxxx[165]='XNA_'
    PTrain_ad.columns=xxxxx
    #split the data
    x=PTrain_ad.iloc[:,1:]
    y=PTrain_ad.iloc[:,0]
    x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.25,random_state=0)
    #%%
    #Brenchmark Model
    #Logistic Regression

    """
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
    y_pred=log.predict(x_val)
    log.coef_
    print("Number of defaults in test set: {0}".format(sum(y_val)))
    print("Number of defaults in train set: {0}".format(sum(y_pred)))
    print(confusion_matrix(y_val,y_pred))
    print(classification_report(y_val,y_pred,digits=3))
    print('The misclassification of kNN:'+'%.3f'%(100*(1-accuracy_score(y_val,y_pred)))+'%')
    """
    #choose the best estimator
    logcv=LogisticRegression(C=1.2589254117941675, class_weight='balanced', dual=False,
              fit_intercept=True, intercept_scaling=1, max_iter=100,
              multi_class='multinomial', n_jobs=1, penalty='l2',
              random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
              warm_start=False)
    log=logcv.fit(x_train,y_train)
    y_pred=log.predict(x_val)
    print("Number of defaults in test set: {0}".format(sum(y_val)))
    print("Number of defaults in train set: {0}".format(sum(y_pred)))
    print(accuracy_score(y_val,y_pred))
    print(confusion_matrix(y_val,y_pred))
    print(classification_report(y_val,y_pred,digits=3))

    #%%decision tree
    """
    para = {
        'min_samples_leaf': [1,5,10,20],
        'max_depth': np.arange(1,30),
    }
    
    dt = GridSearchCV(DecisionTreeClassifier(min_samples_leaf=5,class_weight='balanced'), 
                               para,scoring='f1', n_jobs=-1, cv=5)
    dt.fit(x_tr, y_tr)
    y_pred=dt.predict(x_val)
    print('best score is:',str(dt.best_score_))
    print('Best parameters:',str( dt.best_params_))
    print(confusion_matrix(yz_val,y_pred))
    print(classification_report(yz_val,y_pred))
    """
    #chooose the best estimator
    dt = DecisionTreeClassifier(max_depth= 3, min_samples_leaf= 7)
    dt.fit(x_train, y_train)
    y_pred=dt.predict(x_val)


    print(confusion_matrix(y_val,y_pred))
    print(classification_report(y_val,y_pred,digits=3))
    ##
    #%%
    #NN model
    """
    param_gridknn = {
      'solver': ['lbfgs'], 'max_iter': [500,1000,1500], 'alpha': 10.0 ** -np.arange(1, 7), 'hidden_layer_sizes':np.arange(5, 12), 'random_state':[0,1,2,3,4,5,6,7,8,9]
    }
    NN = GridSearchCV(MLPClassifier(),param_gridknn,cv=10,n_jobs=-1,scoring='f1',verbose=1)
    gridKNN.fit(train_data,train_labels)
    
    NN.fit(x_tr, y_tr)
    y_pred=dt.predict(x_val)
    print('best score is:',str(NN.best_score_))
    print('Best parameters:',str( NN.best_params_))
    print(confusion_matrix(yz_val,y_pred))
    print(classification_report(yz_val,y_pred))
    """
    #choose the best estimator
    NN = MLPClassifier(alpha= 0.1, hidden_layer_sizes= 5, max_iter= 600, random_state= 0, solver='lbfgs')
    #gridKNN.fit(train_data,train_labels)

    NN.fit(x_train, y_train)
    y_pred=NN.predict(x_val)

    print(confusion_matrix(y_val,y_pred))
    print(classification_report(y_val,y_pred))
    #%% RandomForest
    """
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
            warm_start=False),para, scoring='roc_auc',cv=10).fit(x_trains,y_train)  
    yyy=clf.predict(x_tests)
    print(accuracy_score(y_test,yyy))
    print(confusion_matrix(y_test,yyy))
    print(classification_report(y_test,yyy,digits=3))
    print(clf.best_estimator_)
    """
    #choose the best estimator
    clf=RandomForestClassifier(
            bootstrap=True, class_weight='balanced',
                criterion='gini', max_depth=4, max_features=20,
                max_leaf_nodes=None, min_impurity_decrease=0.0,
                min_impurity_split=None, min_samples_leaf=2,
                min_samples_split=8, min_weight_fraction_leaf=0.05,
                n_estimators=120, n_jobs=1, oob_score=True, random_state=0,
                verbose=0, warm_start=False)
    clf=clf.fit(x_train,y_train)
    y_pred_clf=clf.predict(x_val)

    print("Number of defaults in test set: {0}".format(sum(y_val)))
    print("Number of defaults in train set: {0}".format(sum(y_pred_clf)))
    print(accuracy_score(y_val,y_pred_clf))
    print(confusion_matrix(y_val,y_pred_clf))
    print(classification_report(y_val,y_pred_clf,digits=3))

    #%%
    #xgboost model
     #Perforing grid search

    import matplotlib.pylab as plt
    from matplotlib.pylab import rcParams
    rcParams['figure.figsize'] = 45,15
    def XGBFEA(model, xdataset,ydataset,rounds):
        param = model.get_xgb_params()
        xgtrain = xgb.DMatrix(xdataset.values, label=ydataset.values)
        cvresult = xgb.cv(param, xgtrain, num_boost_round=rounds,
                          nfold=5,metrics='f1', early_stopping_rounds=50)
        model.set_params(n_estimators=cvresult.shape[0])
        print("Best n_estimator:"+str(cvresult.shape[0]))
        #Fit the model and predict the data
        model.fit(xdataset, ydataset,eval_metric='f1')
        d_pred = model.predict(xdataset)
        d_pred_prob = model.predict_proba(xdataset)[:,1]
        #Print model report:
        print ("\nModel Report")
        print ("The model Accuracy of training data: %.4g" % metrics.accuracy_score(ydataset.values, d_pred))
        print ("The AUC Score for traning data: %f" % metrics.roc_auc_score(ydataset, d_pred_prob))
        feat_imp = pd.Series(model.get_booster().get_fscore()).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.plot()
        plt.savefig("Feature Importance Score.png")
        return(feat_imp)
    #%% choose the best n_estimator
    """ we Invalidate it 
    xgb_fea = XGBClassifier( learning_rate =0.05, max_depth=5, min_child_weight=1, 
                         subsample=0.8, gamma=0, colsample_bytree=0.8, 
                         objective= 'binary:logistic', nthread=4, scale_pos_weight=1, 
                         seed=27,verbose_eval=20000)
    XGBFEA(xgb_fea, x_train,y_train,1000)
    """
    # the best n_estimator is 1475
    #%%
    """
    if __name__=='__main__':
        param_1 = {'max_depth':[4],#[1,2,3,4,5,6,7,8,9]for gridserach CV,4 is the best estimator,
                   'min_child_weight':[1]#[0,1,2,3,4,5,6,7,8,9]for gridsearch cv, 1 is the best estimator
                   #'Gamma':[0.1,0.2,0.3,0.4],
        }
        xgbcv = GridSearchCV(XGBClassifier(learning_rate =0.05, n_estimators=1475, max_depth=5,
                        min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                         objective= 'binary:logistic', nthread=4, scale_pos_weight=9, seed=27), 
                         param_grid = param_1, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
    xgbcv=xgbcv.fit(x_train,y_train)
    print("grid_scores"+str(xgbcv.grid_scores_))
    print("best_params_:"+str( xgbcv.best_params_))
    print("best_scores:"+str( xgbcv.best_score_))
    yz_pred=xgbcv.predict(x_val)
    print("accuracy_score:"+str(accuracy_score(y_val,yz_pred)))
    print(confusion_matrix(y_val,yz_pred))
    print(classification_report(y_val,yz_pred))
    print(xgbcv.best_estimator_)
    """
    #%%

    #%%
    #Feature selection
    """
    xgb_fea = XGBClassifier( learning_rate =0.3, max_depth=4, min_child_weight=1, 
                         subsample=0.8, gamma=0, colsample_bytree=0.8, 
                         objective= 'binary:logistic', nthread=4, scale_pos_weight=1, 
                         seed=27)
    XGBFEA(xgb_fea, x,y,2000)
    """
    #%%
    #The best estimator
    xgbcv = XGBClassifier(learning_rate =0.05, n_estimators=1475, max_depth=4,reg_alpha=0.01,
                        min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
                         objective= 'binary:logistic', nthread=4, scale_pos_weight=8, seed=27)
    xgbcv=xgbcv.fit(x_train,y_train)
    y_pred=xgbcv.predict(x_val)

    print("accuracy_score:"+str(accuracy_score(y_val,y_pred)))
    print(confusion_matrix(y_val,y_pred))
    print(classification_report(y_val,y_pred,digits=4))
    print(y_pred.sum()/len(y_pred))
    #%%
    """
    Extract 20 important features
    """

    imp_features=['TARGET','SK_ID_CURR','EXT_SOURCE_3', 'DAYS_BIRTH', 'AMT_CREDIT', 'EXT_SOURCE_2', 'AMT_ANNUITY', 'DAYS_ID_PUBLISH', 'DAYS_REGISTRATION', 'DAYS_EMPLOYED', 'DAYS_LAST_PHONE_CHANGE', 'AMT_INCOME_TOTAL' ,'REGION_POPULATION_RELATIVE', 'YEARS_BEGINEXPLUATATION_AVG', 'HOUR_APPR_PROCESS_START', 'FLOORSMAX_AVG', 'AMT_REQ_CREDIT_BUREAU_YEAR' ,'OBS_30_CNT_SOCIAL_CIRCLE','CODE_GENDER' , 'DEF_30_CNT_SOCIAL_CIRCLE','CODE_GENDER','NAME_FAMILY_STATUS','NAME_CONTRACT_TYPE']
    Imxytrain=PTrain_o.loc[:,imp_features].reindex()
    Im_xytrain=pd.concat([Imxytrain.iloc[:,:18],Imxytrain.iloc[:,19:]],axis=1)
    Im_xytrain=Deal_ex(Im_xytrain)
    xim=Im_xytrain.iloc[:,2:]
    yim=Im_xytrain.iloc[:,0]
    xim_train,xim_val,yim_train,yim_val = train_test_split(xim,yim,test_size=0.25,random_state=0)
    x111=Im_xytrain.columns.duplicated(keep = False)
    x222=pd.DataFrame([Im_xytrain.columns,x111],index=['name','value']).T
    x333=x222[x222['value']==True]['name'].reset_index()
    #%%
    #logistic regression

    """
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
    y_pred=log.predict(x_val)
    log.coef_
    print("Number of defaults in test set: {0}".format(sum(y_val)))
    print("Number of defaults in train set: {0}".format(sum(y_pred)))
    print(confusion_matrix(y_val,y_pred))
    print(classification_report(y_val,y_pred,digits=3))
    print('The misclassification of kNN:'+'%.3f'%(100*(1-accuracy_score(y_val,y_pred)))+'%')
    """
    #best estimator
    logcv=LogisticRegression(C=0.3981071705534972, class_weight='balanced', dual=False,
              fit_intercept=True, intercept_scaling=1, max_iter=100,
              multi_class='multinomial', n_jobs=1, penalty='l2',
              random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
              warm_start=False)
    lr=logcv.fit(xim_train,yim_train)
    y_pred=logcv.predict(xim_val)
    print("Number of defaults in test set: {0}".format(sum(yim_val)))
    print("Number of defaults in train set: {0}".format(sum(y_pred)))
    print(accuracy_score(yim_val,y_pred))
    print(confusion_matrix(yim_val,y_pred))
    print(classification_report(yim_val,y_pred,digits=3))


    #decision tree
    #best estimator
    dt = DecisionTreeClassifier(max_depth= 6, min_samples_leaf= 10)

    dt.fit(xim_train, yim_train)
    y_pred=dt.predict(xim_val)
    print(confusion_matrix(yim_val,y_pred))
    #print(classification_report(yim_val,y_pred,digits=3))
    #NN
    #best estimator
    NN = MLPClassifier(alpha= 0.1, hidden_layer_sizes= 5, max_iter= 600, random_state= 0, solver='lbfgs')
    #gridKNN.fit(train_data,train_labels)
    NN.fit(xim_train, yim_train)

    y_pred=NN.predict(xim_val)
    print(confusion_matrix(yim_val,y_pred))
    #print(classification_report(yim_val,y_pred))
    #random forest
    #best estimator
    clf_20=RandomForestClassifier(bootstrap=True, class_weight='balanced',
                criterion='gini', max_depth=4, max_features=20,
                max_leaf_nodes=None, min_impurity_decrease=0.0,
                min_impurity_split=None, min_samples_leaf=2,
                min_samples_split=8, min_weight_fraction_leaf=0.05,
                n_estimators=120, n_jobs=1, oob_score=True, random_state=0,
                verbose=0, warm_start=False)
    clf_20=clf_20.fit(xim_train,yim_train)
    y_pred_rf=clf_20.predict(xim_val)

    print("Number of defaults in test set: {0}".format(sum(yim_val)))
    print("Number of defaults in train set: {0}".format(sum(y_pred_rf)))
    print(accuracy_score(yim_val,y_pred_rf))
    print(confusion_matrix(yim_val,y_pred_rf))
    #print(classification_report(yim_val,y_pred_rf,digits=3))
    #%%
    #merge the 20 important features and the extra features

    ztrain=pd.merge(Imxytrain,PTrainB,how='inner',on='SK_ID_CURR')
    ztrain.to_csv("ztrain.csv")

    #改列名
    ztrain=pd.concat([ztrain.iloc[:,:18],ztrain.iloc[:,19:]],axis=1)
    ztrain=Deal_exx(Deal_B(ztrain))
    ztrain=ztrain.drop(columns=['SK_ID_BUREAU','SK_ID_CURR'])
    xztrain=ztrain.iloc[:,1:]
    xtestlabel=xztrain.columns.tolist()
    yztrain=ztrain.iloc[:,0]
    xz_train,xz_val,yz_train,yz_val = train_test_split(xztrain,yztrain,test_size=0.25,random_state=0)
    x111=ztrain.columns.duplicated(keep = False)
    x222=pd.DataFrame([ztrain.columns,x111],index=['name','value']).T
    x333=x222[x222['value']==True]['name'].reset_index()
    #%%

    #for more quicker compution, we delect the code for gridsearch CV for the final code
    #for 20 features+extra
    #logisticregression
    #choose the best estimator
    logcv_z=LogisticRegression(C=2.51188643150958, class_weight='balanced', dual=False,
              fit_intercept=True, intercept_scaling=1, max_iter=100,
              multi_class='multinomial', n_jobs=1, penalty='l2',
              random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
              warm_start=False)
    lr_z=logcv_z.fit(xz_train,yz_train)
    y_pred_z=lr_z.predict(xz_val)
    print("Number of defaults in test set: {0}".format(sum(yz_val)))
    print("Number of defaults in train set: {0}".format(sum(y_pred_z)))
    print(accuracy_score(yz_val,y_pred_z))
    print(confusion_matrix(yz_val,y_pred_z))
    print(classification_report(yz_val,y_pred_z,digits=3))




    #decision tree
    dt = DecisionTreeClassifier(max_depth= 7, min_samples_leaf= 10)
    dt.fit(xz_train, yz_train)
    y_pred=dt.predict(xz_val)
    print(confusion_matrix(yz_val,y_pred))
    print(classification_report(yz_val,y_pred,digits=3))


    #NN
    NN = MLPClassifier(alpha= 0.1, hidden_layer_sizes= 9, max_iter= 1000, random_state= 0, solver='lbfgs')
    #gridKNN.fit(train_data,train_labels)
    NN.fit(xz_train, yz_train)
    y_pred=NN.predict(xz_val)

    print(confusion_matrix(yz_val,y_pred))
    print(classification_report(yz_val,y_pred))
    #random forest
    clf_z=RandomForestClassifier(bootstrap=True, class_weight='balanced',
                criterion='gini', max_depth=4, max_features=20,
                max_leaf_nodes=None, min_impurity_decrease=0.0,
                min_impurity_split=None, min_samples_leaf=2,
                min_samples_split=8, min_weight_fraction_leaf=0.05,
                n_estimators=130, n_jobs=1, oob_score=True, random_state=0,
                verbose=0, warm_start=False)
    rf_z=clf_z.fit(xz_train,yz_train)
    y_pred_z=rf_z.predict(xz_val)

    print("Number of defaults in test set: {0}".format(sum(yz_val)))
    print("Number of defaults in train set: {0}".format(sum(y_pred_z)))
    print(accuracy_score(yz_val,y_pred_z))
    print(confusion_matrix(yz_val,y_pred_z))
    print(classification_report(yz_val,y_pred_z,digits=3))

    #%% gridsearch
    """
     
    if __name__=='__main__':
        param_1 = {
                    'max_depth':[8],#[1,2,3,4,5,6,7,8,9],
                   'min_child_weight':[1]#[0,1,2,3,4,5,6,7,8,9]
                   #'gamma':[0.1]
        }
        xgbcv = GridSearchCV(XGBClassifier(learning_rate =0.05, n_estimators=36, max_depth=8,
                        min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                         objective= 'binary:logistic', nthread=4, scale_pos_weight=9, seed=27), 
                         param_grid = param_1, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
    xgbcv=xgbcv.fit(xz_train,yz_train)
    print("grid_scores"+str(xgbcv.grid_scores_))
    print("best_params_:"+str( xgbcv.best_params_))
    print("best_scores:"+str( xgbcv.best_score_))
    yz_pred=xgbcv.predict(xz_val)
    print("accuracy_score:"+str(accuracy_score(yz_val,yz_pred)))
    print(confusion_matrix(yz_val,yz_pred))
    print(classification_report(yz_val,yz_pred))
    print(xgbcv.best_estimator_)
    """

    #%%merger the test data
    imp_xfeatures=['Index_ID','EXT_SOURCE_3', 'DAYS_BIRTH', 'AMT_CREDIT', 'EXT_SOURCE_2', 'AMT_ANNUITY', 'DAYS_ID_PUBLISH', 'DAYS_REGISTRATION', 'DAYS_EMPLOYED', 'DAYS_LAST_PHONE_CHANGE', 'AMT_INCOME_TOTAL' ,'REGION_POPULATION_RELATIVE', 'YEARS_BEGINEXPLUATATION_AVG', 'HOUR_APPR_PROCESS_START', 'FLOORSMAX_AVG', 'AMT_REQ_CREDIT_BUREAU_YEAR' ,'OBS_30_CNT_SOCIAL_CIRCLE','CODE_GENDER' , 'CODE_GENDER','DEF_30_CNT_SOCIAL_CIRCLE','NAME_FAMILY_STATUS','NAME_CONTRACT_TYPE']
    Imxytest=PTest.loc[:,imp_xfeatures].reindex()
    zztest=pd.merge(Imxytest,PTestB,how='inner',on='Index_ID')
    zztest=pd.concat([zztest.iloc[:,:18],zztest.iloc[:,19:]],axis=1)
    zztest=Deal_exx(Deal_B(zztest))
    zztest=zztest.drop(columns=['Index_ID','SK_ID_BUREAU','Bad debt'])
    zztest['Unknown']=0
    zztest['XNA']=0
    zztest['currency 3']=0
    zztest['currency 4']=0
    zztest=zztest.reindex(columns=['EXT_SOURCE_3', 'DAYS_BIRTH', 'AMT_CREDIT', 'EXT_SOURCE_2',
           'AMT_ANNUITY', 'DAYS_ID_PUBLISH', 'DAYS_REGISTRATION', 'DAYS_EMPLOYED',
           'DAYS_LAST_PHONE_CHANGE', 'AMT_INCOME_TOTAL',
           'REGION_POPULATION_RELATIVE', 'YEARS_BEGINEXPLUATATION_AVG',
           'HOUR_APPR_PROCESS_START', 'FLOORSMAX_AVG',
           'AMT_REQ_CREDIT_BUREAU_YEAR', 'OBS_30_CNT_SOCIAL_CIRCLE',
           'DEF_30_CNT_SOCIAL_CIRCLE', 'DAYS_CREDIT', 'CREDIT_DAY_OVERDUE',
           'DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT', 'AMT_CREDIT_MAX_OVERDUE',
           'CNT_CREDIT_PROLONG', 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT',
           'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE', 'DAYS_CREDIT_UPDATE',
           'Closed', 'Sold', 'currency 2', 'currency 3', 'currency 4', 'Car loan',
           'Consumer credit', 'Credit card', 'Loan for business development',
           'Loan for working capital replenishment', 'Microloan', 'Mortgage',
           'Unknown type of loan', 'M', 'XNA', 'Married', 'Separated',
           'Single / not married', 'Unknown', 'Widow', 'Revolving loans'])

    #x111=zztest.columns.duplicated(keep = False)
    #x222=pd.DataFrame([zztest.columns,x111],index=['name','value']).T
    #x333=x222[x222['value']==True]['name'].reset_index()
    #%%
    #Finally, we select Xgboost for our final model and here is the process for the final prediction
    #%%
    """
    xgb_fea = XGBClassifier( learning_rate =0.05, max_depth=5, min_child_weight=1, 
                         subsample=0.8, gamma=0, colsample_bytree=0.8, 
                         objective= 'binary:logistic', nthread=4, scale_pos_weight=1, 
                         seed=27,verbose_eval=20000)
    
    XGBFEA(xgb_fea, xztrain,yztrain,1000)
    """
    #get best n_estimator is 318
    #%%tune max depth
    #I Invalidate the code of tuning
    """
    if __name__=='__main__':
        param_1 = {'max_depth':[9],#[5,6,7,8,9,10,11,12],
    
        }
        xgbcv = GridSearchCV(XGBClassifier(learning_rate =0.03, n_estimators=318, max_depth=9,
                        min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                         objective= 'binary:logistic', nthread=4, scale_pos_weight=10, seed=27,verbose_eval=20000), 
                         param_grid = param_1, scoring='f1',n_jobs=1,iid=False, cv=5,verbose=True)
    xgbcv=xgbcv.fit(xztrain,yztrain)
    y_pred=xgbcv.predict(zztest)
    
    print("grid_scores"+str(xgbcv.grid_scores_))
    print("best_params_:"+str( xgbcv.best_params_))
    print("best_scores:"+str( xgbcv.best_score_))
    print(xgbcv.best_estimator_)
    print(yztrain.sum()/len(yztrain))
    print(y_pred.sum()/len(y_pred))
    #get the best paramter for max_depth=8
    #the scale_pos_weight
    if __name__=='__main__':
        param_1 = {'scale_pos_weight':[11]#[8,10,11,12,13],
    
        }
        xgbcv = GridSearchCV(XGBClassifier(learning_rate =0.03, n_estimators=318, max_depth=9,
                        min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                         objective= 'binary:logistic', nthread=4, scale_pos_weight=11, seed=27,verbose_eval=20000), 
                         param_grid = param_1, scoring='f1',n_jobs=1,iid=False, cv=5,verbose=True)
    xgbcv=xgbcv.fit(xztrain,yztrain)
    y_pred=xgbcv.predict(zztest)
    
    print("grid_scores"+str(xgbcv.grid_scores_))
    print("best_params_:"+str( xgbcv.best_params_))
    print("best_scores:"+str( xgbcv.best_score_))
    print(xgbcv.best_estimator_)
    print(yztrain.sum()/len(yztrain))
    print(y_pred.sum()/len(y_pred))
    #get the best scale_pos_weight =11
    
    
        param_1 = {'min_child_weight':[1]#[1,2,3,4,5]
        }
        xgbcv = GridSearchCV(XGBClassifier(learning_rate =0.03, n_estimators=318, max_depth=9,
                        min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                         objective= 'binary:logistic', nthread=4, scale_pos_weight=11, seed=27,verbose_eval=20000), 
                         param_grid = param_1, scoring='f1',n_jobs=1,iid=False, cv=5,verbose=True)
    xgbcv=xgbcv.fit(xztrain,yztrain)
    y_pred=xgbcv.predict(zztest)
    
    print("grid_scores"+str(xgbcv.grid_scores_))
    print("best_params_:"+str( xgbcv.best_params_))
    print("best_scores:"+str( xgbcv.best_score_))
    print(xgbcv.best_estimator_)
    print(yztrain.sum()/len(yztrain))
    print(y_pred.sum()/len(y_pred))
    #get the best min_child_weight =1
    
    if __name__=='__main__':
        param_1 = {'gamma':[0]#[0,0.1,0.2,0.3]
        }
        xgbcv = GridSearchCV(XGBClassifier(learning_rate =0.03, n_estimators=318, max_depth=8,
                        min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                         objective= 'binary:logistic', nthread=4, scale_pos_weight=9, seed=27,verbose_eval=20000), 
                         param_grid = param_1, scoring='f1',n_jobs=1,iid=False, cv=5,verbose=True)
    xgbcv=xgbcv.fit(xztrain,yztrain)
    y_pred=xgbcv.predict(zztest)
    
    print("grid_scores"+str(xgbcv.grid_scores_))
    print("best_params_:"+str( xgbcv.best_params_))
    print("best_scores:"+str( xgbcv.best_score_))
    print(xgbcv.best_estimator_)
    print(yztrain.sum()/len(yztrain))
    print(y_pred.sum()/len(y_pred))
    #get the best gama=0.1
    """
    #%%

    #The best estimator
    xgbcv = XGBClassifier(learning_rate =0.03, n_estimators=318, max_depth=8,reg_alpha=0.01,
                        min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
                         objective= 'binary:logistic', nthread=4, scale_pos_weight=11, seed=27)
    xgbcv=xgbcv.fit(xz_train,yz_train)
    y_pred=xgbcv.predict(xz_val)

    print("accuracy_score:"+str(accuracy_score(yz_val,y_pred)))
    print(confusion_matrix(yz_val,y_pred))
    print(classification_report(yz_val,y_pred,digits=4))
    print(y_pred.sum()/len(y_pred))
    #%%
    """
    The final forecast
    """

    xgbcv = XGBClassifier(learning_rate =0.03, n_estimators=318, max_depth=8,reg_alpha=0.01,
                        min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
                         objective= 'binary:logistic', nthread=4, scale_pos_weight=11, seed=27)
    xgbcv=xgbcv.fit(xztrain,yztrain)

    yz_pred=xgbcv.predict(zztest)
    print(yz_pred.sum())
    print(yz_pred.sum()/len(yz_pred))
    result=pd.DataFrame({'Index_ID':PTestB['Index_ID'],'TARGET':yz_pred}).to_csv('Group16_Results_Bureau.csv',index=False)
    #%%

