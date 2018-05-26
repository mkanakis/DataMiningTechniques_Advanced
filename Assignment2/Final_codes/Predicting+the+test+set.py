# -*- coding: utf-8 -*-
"""
Created on Wed May 23 10:43:43 2018

@author: MICK
"""

import pandas as pd
import numpy as np

adf = pd.read_csv(r'F:\Vu Datamining files\Correct_column_order.csv')
columns = list(adf['columns'])
columns.remove('relevance')
del adf

df = pd.read_csv(r'F:\Vu Datamining files\Vu_Testset_complete.csv')
df = df.drop('Unnamed: 0', 1)

MODEL_LOCATION = r'F:\Vu Datamining files\modelshows5.sav'
PREDICTIONS_SAVE_LOCATION = r'F:\Vu Datamining files\Group80_Predictions.csv'
# In[6]:

pt = pd.pivot_table(df, values='site_id', index='prop_id', aggfunc='count').reset_index()
pt.columns = ['prop_id', 'show_count']
df = df.merge(pt, how='left', on='prop_id')
del pt

pt = pd.pivot_table(df, values='show_count', index='srch_id', aggfunc='mean').reset_index()
pt.columns = ['srch_id', 'srchid_mean_show_count']
df = df.merge(pt, how='left', on='srch_id')
del pt

df.loc[:, 'abs_diff_srchid_mean_show_count'] = abs(df['srchid_mean_show_count'] - df['show_count'])
df.loc[:, 'true_diff_srchid_mean_show_count'] = df['show_count'] - df['srchid_mean_show_count'] 
df.loc[:, 'per_diff_srchid_mean_show_count'] = (100*(df['show_count'] - df['srchid_mean_show_count']))/df['srchid_mean_show_count']
df = df.drop('srchid_mean_show_count', 1)

df = df[columns]


import pickle

model = pickle.load(open(MODEL_LOCATION, 'rb'))
prediction = model.predict(df.drop('srch_id', 1))

# In[7]:

result = pd.DataFrame()
result.loc[:, 'srch_id'] = df['srch_id']
result.loc[:, 'prop_id'] = df['prop_id']
result.loc[:, 'prediction'] = prediction


# In[11]:

result = result.sort_values(['srch_id', 'prediction'], ascending=[True, False])

handin = result[['srch_id', 'prop_id']]
handin.columns = ['SearchId', 'PropertyId']
handin.to_csv(PREDICTIONS_SAVE_LOCATION, index=False)
