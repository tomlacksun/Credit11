# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 02:35:34 2018

@author: Yhi
"""

#Data
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 13:51:46 2018

@author: Yhi
"""
from scipy import stats
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
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score
PTrain_o=pd.read_csv('ProjectTrain.csv')
#%%
PTrainB=pd.read_csv('ProjectTrain_Bureau.csv').rename(columns={'AMT_ANNUITY':'AMT_ANNUITY_B'})
PTest=pd.read_csv('ProjectTest.csv')
PTestB=pd.read_csv('ProjectTest_Bureau.csv').rename(columns={'AMT_ANNUITY':'AMT_ANNUITY_B'})
#ztest=pd.merge(PTest,PTestB,how='inner',on='Index_ID').iloc[:,1:]

#%%
PTrain=PTrain_o.iloc[:,1:]
PTrain.head()
PTrain.describe()
corr=PTrain.corr().abs()
corr_ad=corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
to_drop = [column for column in corr_ad.columns if any(corr_ad[column] > 0.8)]
print('%d columns with more than 0.8 correaltion is removed.' % (len(to_drop)))
PTrain = PTrain.drop(columns = to_drop)
miss = PTrain.isnull().sum()
missper = (PTrain.isnull().sum()/len(PTrain))
missing_data = pd.DataFrame({'Total':miss,'Percent':missper}).reset_index().sort_values(by='Total',ascending=False)
missing11=missing_data[missing_data['Percent']>0.5]['index']
print('%d columns with more than 50%% missing values is removed.' % len(missing11))
PTrain = PTrain.drop(columns = missing11)


def Clean(dataset):
    dataset=dataset.drop(columns = to_drop)
    dataset=dataset.drop(columns = missing11)
    return dataset
def Deal(dataset):
    col_mean=['AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR','AMT_ANNUITY','EXT_SOURCE_2','EXT_SOURCE_3','YEARS_BEGINEXPLUATATION_AVG','FLOORSMAX_AVG',]
    col_mode=['OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE' ,'AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON']
    col_log=['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY']
    col_dum=['NAME_CONTRACT_TYPE','CODE_GENDER', 'FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE','WEEKDAY_APPR_PROCESS_START','ORGANIZATION_TYPE','EMERGENCYSTATE_MODE']
    for i in col_mean:
        dataset.loc[:,i]=dataset.loc[:,i].fillna(dataset.loc[:,i].mean())
    for i in col_mode:
        dataset.loc[:,i]=dataset.loc[:,i].fillna(stats.mode( dataset.loc[:,i])[0][0])    
    for i in col_log:
        dataset.loc[:,i]=np.log1p(dataset.loc[:,i])
    for i in col_dum:
        dataset=pd.concat([dataset,pd.get_dummies(dataset.loc[:,i],drop_first=True)],axis=1)
        dataset=dataset.drop(columns=i)
    return dataset
PTrain_ad=Deal(PTrain)

miss = PTrain.isnull().sum()

def Deal_B(dataset):
    col_log=['AMT_CREDIT_MAX_OVERDUE','AMT_CREDIT_SUM','AMT_CREDIT_SUM_DEBT','AMT_CREDIT_SUM_LIMIT','AMT_CREDIT_SUM_OVERDUE']
    col_dum=['CREDIT_ACTIVE','CREDIT_CURRENCY','CREDIT_TYPE']
    col_zero=['DAYS_CREDIT_ENDDATE','DAYS_ENDDATE_FACT']
    for i in col_log:
        dataset.loc[:,i]=dataset.loc[:,i].fillna(0)
        dataset.loc[:,i]=np.log1p(dataset.loc[:,i]).fillna(0)
    for i in col_dum:
        dataset=pd.concat([dataset,pd.get_dummies(dataset.loc[:,i],drop_first=True)],axis=1)
        dataset=dataset.drop(columns=i)
    for i in col_zero:
        dataset.loc[:,i]=dataset.loc[:,i].fillna(0)
    dataset=dataset.drop(columns='AMT_ANNUITY_B')
    return dataset
#%%
#ztest=Clean(ztest)
#ztest=Deal_B(ztest)

#ztest=Deal(ztest)


#%%
#改列名
x111=PTrain_ad.columns.duplicated(keep = False)
x222=pd.DataFrame([PTrain_ad.columns,x111],index=['name','value']).T
x333=x222[x222['value']==True]['name'].reset_index()
xxxxx=list(PTrain_ad.columns)
xxxxx[58]='Y_'
xxxxx[165]='XNA_'
PTrain_ad.columns=xxxxx
#%%Train 的数据
x=PTrain_ad.iloc[:,1:]
y=PTrain_ad.iloc[:,0]
x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.25,random_state=0)
x.to_csv("qqq.csv")



#%%
"""
抽出最重要的20 
"""
def Deal_ex(dataset):
    col_mean=['AMT_ANNUITY','EXT_SOURCE_2','AMT_REQ_CREDIT_BUREAU_YEAR','EXT_SOURCE_3','YEARS_BEGINEXPLUATATION_AVG','FLOORSMAX_AVG']
    col_mode=['OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE']
    col_log=['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY']
    col_dum=['CODE_GENDER','NAME_FAMILY_STATUS','NAME_CONTRACT_TYPE']
   
    for i in col_mean:
        dataset.loc[:,i]=dataset.loc[:,i].fillna(dataset.loc[:,i].mean())
    for i in col_mode:
        dataset.loc[:,i]=dataset.loc[:,i].fillna(stats.mode( dataset.loc[:,i])[0][0])    
    for i in col_log:
        dataset.loc[:,i]=np.log1p(dataset.loc[:,i])

    for i in col_dum:
        dataset=pd.concat([dataset,pd.get_dummies(dataset.loc[:,i],drop_first=True)],axis=1)
        dataset=dataset.drop(columns=i)
    return dataset
imp_features=['TARGET','SK_ID_CURR','EXT_SOURCE_3', 'DAYS_BIRTH', 'AMT_CREDIT', 'EXT_SOURCE_2', 'AMT_ANNUITY', 'DAYS_ID_PUBLISH', 'DAYS_REGISTRATION', 'DAYS_EMPLOYED', 'DAYS_LAST_PHONE_CHANGE', 'AMT_INCOME_TOTAL' ,'REGION_POPULATION_RELATIVE', 'YEARS_BEGINEXPLUATATION_AVG', 'HOUR_APPR_PROCESS_START', 'FLOORSMAX_AVG', 'AMT_REQ_CREDIT_BUREAU_YEAR' ,'OBS_30_CNT_SOCIAL_CIRCLE','CODE_GENDER' , 'DEF_30_CNT_SOCIAL_CIRCLE','CODE_GENDER','NAME_FAMILY_STATUS','NAME_CONTRACT_TYPE']
Imxytrain=PTrain_o.loc[:,imp_features].reindex()
Im_xytrain=pd.concat([Imxytrain.iloc[:,:18],Imxytrain.iloc[:,19:]],axis=1)

Im_xytrain=Deal_ex(Im_xytrain)
xim=Im_xytrain.iloc[:,2:]
yim=Im_xytrain.iloc[:,0]
xim_train,xim_val,yim_train,yim_val = train_test_split(xim,yim,test_size=0.25,random_state=0)
#这里随便做一下三个模型和之前所有数据的对比，比f1 socre matrix 图之类的 
x111=Im_xytrain.columns.duplicated(keep = False)
x222=pd.DataFrame([Im_xytrain.columns,x111],index=['name','value']).T
x333=x222[x222['value']==True]['name'].reset_index()

#%%
"""
抽出最重要的20 和extra合并
"""

ztrain=pd.merge(Imxytrain,PTrainB,how='inner',on='SK_ID_CURR')
ztrain.to_csv("ztrain.csv")

#改列名
ztrain=pd.concat([ztrain.iloc[:,:18],ztrain.iloc[:,19:]],axis=1)
ztrain=Deal_ex(Deal_B(ztrain))
ztrain=ztrain.drop(columns=['SK_ID_BUREAU','SK_ID_CURR'])
xztrain=ztrain.iloc[:,1:]
yztrain=ztrain.iloc[:,0]
xz_train,xz_val,yz_train,yz_val = train_test_split(xztrain,yztrain,test_size=0.25,random_state=0)
x111=ztrain.columns.duplicated(keep = False)
x222=pd.DataFrame([ztrain.columns,x111],index=['name','value']).T
x333=x222[x222['value']==True]['name'].reset_index()



#%%合并test
imp_xfeatures=['Index_ID','EXT_SOURCE_3', 'DAYS_BIRTH', 'AMT_CREDIT', 'EXT_SOURCE_2', 'AMT_ANNUITY', 'DAYS_ID_PUBLISH', 'DAYS_REGISTRATION', 'DAYS_EMPLOYED', 'DAYS_LAST_PHONE_CHANGE', 'AMT_INCOME_TOTAL' ,'REGION_POPULATION_RELATIVE', 'YEARS_BEGINEXPLUATATION_AVG', 'HOUR_APPR_PROCESS_START', 'FLOORSMAX_AVG', 'AMT_REQ_CREDIT_BUREAU_YEAR' ,'OBS_30_CNT_SOCIAL_CIRCLE','CODE_GENDER' , 'CODE_GENDER','DEF_30_CNT_SOCIAL_CIRCLE','NAME_FAMILY_STATUS','NAME_CONTRACT_TYPE']

Imxytest=PTest.loc[:,imp_xfeatures].reindex()

zztest=pd.merge(Imxytest,PTestB,how='inner',on='Index_ID')
zztest=pd.concat([zztest.iloc[:,:18],zztest.iloc[:,19:]],axis=1)
zztest=Deal_ex(Deal_B(zztest))
zztest=zztest.drop(columns=['Index_ID','SK_ID_BUREAU','Bad debt'])
zztest['Unknown']=0
zztest['XNA']=0
zztest['currency 3']=0
zztest['currency 4']=0
#x111=zztest.columns.duplicated(keep = False)
#x222=pd.DataFrame([zztest.columns,x111],index=['name','value']).T
#x333=x222[x222['value']==True]['name'].reset_index()
#%%

