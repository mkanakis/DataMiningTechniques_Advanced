
# coding: utf-8

# In[1]:


from pyltr.models import LambdaMART
from pyltr.metrics import NDCG
from pyltr.models.monitors import ValidationMonitor
import pandas as pd
import numpy as np

df = pd.read_csv(r'F:\Vu_Training_dataset.csv')

df = df.drop(['click_bool', 'booking_bool', 'position'], 1)
for column in ['booking_season', 'booking_month', 'booking_week', 'booking_weekday',
       'booking_hour', 'incheck_season', 'incheck_month', 'incheck_week',
       'incheck_weekday', 'checkout_season', 'checkout_month', 'checkout_week',
       'checkout_weekday', 'site_id', 'visitor_location_country_id', 'prop_country_id', 
        'srch_destination_id']:
    df.loc[:, column] = df[column].astype('category')
del column

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
# In[2]:

from sklearn.model_selection import train_test_split as tts

TRAINSIZE = 40000
TESTSIZE = 90000
VALSIZE = 10000

srch_ids = list(set(df['srch_id']))

x_val_ids, x_traintest_ids, y_val_ids, y_traintest_ids = tts(srch_ids, srch_ids, test_size = .9)
del y_traintest_ids, y_val_ids

x_train_ids, x_test_ids, y_train_ids, y_test_ids = tts(x_traintest_ids, x_traintest_ids, test_size = .5)
del y_train_ids, y_test_ids

#Train: n = 44954 , Val: n = 9989  , Test: n = 44955

x_train = df[df['srch_id'].isin(x_train_ids[:TRAINSIZE])].drop(['srch_id', 'relevance'], 1)
y_train = df[df['srch_id'].isin(x_train_ids[:TRAINSIZE])]['relevance']
qids_train = df[df['srch_id'].isin(x_train_ids[:TRAINSIZE])]['srch_id']

x_val = df[df['srch_id'].isin(x_val_ids[:VALSIZE])].drop(['srch_id', 'relevance'], 1)
y_val = df[df['srch_id'].isin(x_val_ids[:VALSIZE])]['relevance']
qids_val = df[df['srch_id'].isin(x_val_ids[:VALSIZE])]['srch_id']

x_test = df[df['srch_id'].isin(x_test_ids[:TESTSIZE])].drop(['srch_id', 'relevance'], 1)
y_test = df[df['srch_id'].isin(x_test_ids[:TESTSIZE])]['relevance']
qids_test = df[df['srch_id'].isin(x_test_ids[:TESTSIZE])]['srch_id']


# In[3]:


y_train = np.asarray(y_train)
qids_train = np.asarray(qids_train)
y_val = np.asarray(y_val)
qids_val = np.asarray(qids_val)
y_test = np.asarray(y_test)
qids_test = np.asarray(qids_test)


# In[4]:


metric = NDCG(k=40)

monitor = ValidationMonitor(x_val, y_val, qids_val, metric=metric, stop_after=50)


# In[5]:


model = LambdaMART(metric=metric, max_depth = 4, n_estimators=450, learning_rate=.04, verbose=1, max_features = 25,
                  min_samples_split = 1000, min_samples_leaf = 200, max_leaf_nodes = 20)
model.fit(x_train, y_train, qids_train, monitor=monitor)


# In[ ]:


prediction = model.predict(x_test)

print ('Random ranking:', metric.calc_mean_random(qids_test, y_test))
print ('Our model:', metric.calc_mean(np.asarray(qids_test), np.asarray(y_test), prediction))

prediction_train = model.predict(x_train)
print ('Train model:', metric.calc_mean(np.asarray(qids_train), np.asarray(y_train), prediction_train))


# In[ ]:


#a = zip(model.feature_importances_, x_train.columns)
#b = list()
#for imp, name in a:
#   b.append((imp, name))
#sorted(b)


# In[ ]:


import pickle
pickle.dump(model, open(r'C:\Users\MICK\Downloads\modelshows5.sav', 'wb'))

# In[ ]:


#result = pd.DataFrame()
#result.loc[:, 'srch_id'] = np.asarray(qids_test)
#result.loc[:, 'prop_id'] = (x_test['prop_id']).reset_index()['prop_id']
#result.loc[:, 'prediction'] = prediction
#result.loc[:, 'relevance'] = np.asarray(y_test)


# In[ ]:


#result.sort_values(['srch_id', 'prediction'], ascending=[True, False])


# In[ ]:


