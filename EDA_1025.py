#!/usr/bin/env python
# coding: utf-8

# In[6]:


#import relevant packages
import numpy as np
np.set_printoptions(threshold=np.inf) 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mode
from scipy.stats import norm,skew,kurtosis,stats
from pandas.plotting import scatter_matrix


# In[7]:


#read csv data
data=pd.read_csv('ProjectTrain.csv')

#check data types
data.dtypes

data.get_dtype_counts()

#check missing vlaue
data.count()

data.describe()


#top missing words
data.isnull().sum().sort_values(ascending=False)

#%%
correlations = data.corr()['TARGET'].sort_values()
#print('Most Positive Correlations: \n', correlations.tail(15))
#print('\nMost Negative Correlations: \n', correlations.head(15))



#examine the correlations between variables
corr=data.corr().round(3)
#print(corr)

data.corr().unstack().sort_values().drop_duplicates()


# Generate a custom diverging colormap
plt.figure(figsize=(16,10))
sns.heatmap(corr,annot=True, fmt=".2f")
plt.plot()

#Distribution of data
plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_INCOME_TOTAL")
sns.distplot(data['AMT_INCOME_TOTAL'])

plt.plot()

plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_CREDIT")
sns.distplot(data['AMT_CREDIT'].dropna())


plt.plot()
# In[9]:


plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_ANNUITY")
sns.distplot(data['AMT_ANNUITY'].dropna())
plt.plot()

# In[10]:


plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_GOODS_PRICE")
sns.distplot(data['AMT_GOODS_PRICE'].dropna())
plt.plot()

# 'NAME_HOUSING_TYPE' What is the housing situation of the client (renting, living with parents, ...)


temp = data["NAME_HOUSING_TYPE"].value_counts()
labels = temp.index
sizes = temp.values
plt.figure(figsize=(8,8))
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%')
ax1.axis('equal')
plt.plot()



temp = data["CODE_GENDER"].value_counts()
labels = temp.index
sizes = temp.values
plt.figure(figsize=(8,8))
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%')
ax1.axis('equal')
plt.plot()


#%%


plt.figure(figsize=(15,5))
sns.countplot(x="OWN_CAR_AGE", data=data)  

plt.plot()

plt.figure(figsize=(15,5))
sns.countplot(x="OCCUPATION_TYPE", data=data)
plt.plot()

plt.figure(figsize=(15,5))
sns.countplot(x="CNT_FAM_MEMBERS", data=data)
plt.plot()

plt.figure(figsize=(15,5))
sns.countplot(x="EXT_SOURCE_3", data=data)
plt.plot()

plt.figure(figsize=(15,5))
sns.countplot(x="OBS_30_CNT_SOCIAL_CIRCLE", data=data)
plt.plot()

sns.countplot(x="AMT_REQ_CREDIT_BUREAU_HOUR", data=data)
plt.plot()



sns.countplot(x="AMT_REQ_CREDIT_BUREAU_WEEK", data=data)
plt.plot()

# Plot the distribution of ages in years
plt.hist(abs(data['DAYS_BIRTH'] / 365), edgecolor = 'k')
plt.title('Age of Client'); plt.xlabel('Age (years)'); plt.ylabel('Count');
plt.plot()


sns.kdeplot(data.loc[data['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'target == 0')
sns.kdeplot(data.loc[data['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target == 1')
plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages')          


