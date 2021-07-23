#!/usr/bin/env python
# coding: utf-8

# In[101]:



# import packages
import os
import joblib
import numpy as np
import pandas as pd
import warnings
import csv
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from matplotlib import ticker
from sklearn.preprocessing import OneHotEncoder

from skopt import BayesSearchCV
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold


# In[102]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')


# In[103]:


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

train_df['date_time'] = pd.to_datetime(train_df['date_time'])
test_df['date_time'] = pd.to_datetime(test_df['date_time'])


# In[104]:


columns = test_df.columns[1:]

X = train_df[columns].values
X_test = test_df[columns].values
target_0 = train_df['target_carbon_monoxide'].values.reshape(-1,1)
target_1 = train_df['target_benzene'].values.reshape(-1,1)
target_2 = train_df['target_nitrogen_oxides'].values.reshape(-1,1)

train_oof = np.zeros((train_df.shape[0],3))
test_preds = np.zeros((test_df.shape[0],3))


# In[111]:



n_splits = 5
ALPHA = 3500

kf = KFold(n_splits=n_splits, random_state=137, shuffle=True)
# print(enumerate(kf.split(train_df)))
for jj, (train_index, val_index) in enumerate(kf.split(train_df)):
    print("Fitting fold", jj+1)
    train_features = X[train_index]
    train_target_0 = target_0[train_index]
    train_target_1 = target_1[train_index]
    train_target_2 = target_2[train_index]
    
    val_features = X[val_index]
    val_target_0 = target_0[val_index]
    val_target_1 = target_1[val_index]
    val_target_2 = target_2[val_index]
    
    model = Ridge(alpha=ALPHA)
    model.fit(train_features, train_target_0)
    val_pred_0 = model.predict(val_features)
    train_oof[val_index,0] = val_pred_0.flatten()
    test_preds[:,0] += model.predict(X_test).flatten()/n_splits
    
    model = Ridge(alpha=ALPHA)
    model.fit(train_features, train_target_1)
    val_pred_1 = model.predict(val_features)
    train_oof[val_index,1] = val_pred_1.flatten()
    test_preds[:,1] += model.predict(X_test).flatten()/n_splits
    
    model = Ridge(alpha=ALPHA)
    model.fit(train_features, train_target_2)
    val_pred_2 = model.predict(val_features)
    train_oof[val_index,2] = val_pred_2.flatten()
    test_preds[:,2] += model.predict(X_test).flatten()/n_splits


# In[112]:


np.save('train_oof', train_oof)
np.save('test_preds', test_preds)
target = np.hstack([target_0, target_1, target_2])

submission[submission.columns[1:]] = test_preds
submission.to_csv('submissionB.csv', index=False)


# In[ ]:





# In[ ]:




