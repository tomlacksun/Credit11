# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 01:41:42 2018

@author: Yhi
"""

import pandas as pd
import numpy as np

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

def Deal_ex(dataset):
    col_mean=['AMT_REQ_CREDIT_BUREAU_YEAR' ,'AMT_ANNUITY','EXT_SOURCE_2','EXT_SOURCE_3','YEARS_BEGINEXPLUATATION_AVG','FLOORSMAX_AVG']
    col_mode=['OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE']
    #col_log=['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY']
    col_dum=['CODE_GENDER','NAME_FAMILY_STATUS','NAME_CONTRACT_TYPE']
    for i in col_mean:
        dataset.loc[:,i]=dataset.loc[:,i].fillna(dataset.loc[:,i].mean())
    for i in col_mode:
        dataset.loc[:,i]=dataset.loc[:,i].fillna(stats.mode( dataset.loc[:,i])[0][0])
    #for i in col_log:
        #dataset.loc[:,i]=np.log1p(dataset.loc[:,i])
    for i in col_dum:
        dataset=pd.concat([dataset,pd.get_dummies(dataset.loc[:,i],drop_first=True)],axis=1)
        dataset=dataset.drop(columns=i)
    return dataset


def Deal_exx(dataset):
    col_mean=['AMT_ANNUITY','EXT_SOURCE_2','EXT_SOURCE_3','YEARS_BEGINEXPLUATATION_AVG','FLOORSMAX_AVG']
    col_mode=['OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE']
    col_log=['EXT_SOURCE_3', 'DAYS_BIRTH', 'EXT_SOURCE_2', 'DAYS_ID_PUBLISH', 'DAYS_REGISTRATION', 'DAYS_EMPLOYED', 'DAYS_LAST_PHONE_CHANGE', 'REGION_POPULATION_RELATIVE', 'YEARS_BEGINEXPLUATATION_AVG', 'FLOORSMAX_AVG']
    col_dum=['CODE_GENDER','NAME_FAMILY_STATUS','NAME_CONTRACT_TYPE']
    for i in col_mean:
        dataset.loc[:,i]=dataset.loc[:,i].fillna(dataset.loc[:,i].mean())
    for i in col_mode:
        dataset.loc[:,i]=dataset.loc[:,i].fillna(stats.mode( dataset.loc[:,i])[0][0])
    for i in col_log:
        dataset.loc[:,i]=np.log1p(abs(dataset.loc[:,i]))
    for i in col_dum:
        dataset=pd.concat([dataset,pd.get_dummies(dataset.loc[:,i],drop_first=True)],axis=1)
        dataset=dataset.drop(columns=i)
    return dataset
