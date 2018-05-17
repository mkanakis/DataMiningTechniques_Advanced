
# coding: utf-8

# In[1]:


from pyltr.models import LambdaMART
from pyltr.metrics import NDCG
from pyltr.models.monitors import ValidationMonitor
import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('Dataset_test.csv')
for column in ['booking_season', 'booking_month', 'booking_week', 'booking_weekday',
       'booking_hour', 'incheck_season', 'incheck_month', 'incheck_week',
       'incheck_weekday', 'checkout_season', 'checkout_month', 'checkout_week',
       'checkout_weekday']:
    df.loc[:, column] = df[column].astype('category')


# In[3]:


srch_ids = list(set(df['srch_id']))


# In[4]:


x_train = df[df['srch_id'].isin(srch_ids[20000:30000])].drop(['Unnamed: 0', 'srch_id', 'relevance'], 1)
y_train = df[df['srch_id'].isin(srch_ids[20000:30000])]['relevance']
qids_train = df[df['srch_id'].isin(srch_ids[20000:30000])]['srch_id']

x_val = df[df['srch_id'].isin(srch_ids[10000:13000])].drop(['Unnamed: 0', 'srch_id', 'relevance'], 1)
y_val = df[df['srch_id'].isin(srch_ids[10000:13000])]['relevance']
qids_val = df[df['srch_id'].isin(srch_ids[10000:13000])]['srch_id']

x_test = df[df['srch_id'].isin(srch_ids[:10000])].drop(['Unnamed: 0', 'srch_id', 'relevance'], 1)
y_test = df[df['srch_id'].isin(srch_ids[:10000])]['relevance']
qids_test = df[df['srch_id'].isin(srch_ids[:10000])]['srch_id']


# In[5]:


metric = NDCG(k=40)

monitor = ValidationMonitor(x_val, y_val, qids_val, metric=metric, stop_after=50)


# In[6]:


model = LambdaMART(metric=metric, max_depth = 6, n_estimators=100, learning_rate=.1, verbose=1)
model.fit(x_train, y_train, qids_train, monitor=monitor)


# In[7]:


prediction = model.predict(x_test)

print ('Random ranking:', metric.calc_mean_random(qids_test, y_test))
print ('Our model:', metric.calc_mean(np.asarray(qids_test), np.asarray(y_test), prediction))

prediction_train = model.predict(x_train)
print ('Train model:', metric.calc_mean(np.asarray(qids_train), np.asarray(y_train), prediction_train))


# The training dataset contains some variables that dont exist in the test set. So in the test set we have to impute those values based on the values that are derived from the training set.
# 
#     e.a. the mean_click/mean_booking of a property cant be computed directly from the test set, so we have to impute the mean_click/mean_booking from the same property in the training set. This means that these attributes contain less information for the test set, since they arent computed based on the test set. When we train a model on the training set the model will assign a high importance to these variables since they correlate strongly with the relevance score. This can distort the accuracy of the predictions on the test set.
#     
#         Proof:
#             The correlations between mean_click/mean_booking/position and relevance in the train_train set:
#                 click: r = .243
#                 book: r = .269
#                 position: r = -.107
#             
#             The correlations between mean_click/mean_booking/position and relevance in the train_test set:
#                 click: r = .091
#                 book: r = .093
#                 position: r = -.095
#     
# To solve this i split the training set into two parts (train_train and train_test). The goal is to use the train_train set in the same way we would use the train set and the train_test set in the same way we would use the test set.
#     
#     This means that the click_bool/booking_bool/position variables that are used to engineer attributes will not be used in the train_test set, similar to how we would engineer the features for the test set. Therefore the train_test set will be extremely comparable to the actual test set. So if we train, validate and test a model on the train_test set it would perform the same as it would perform on the test set.
#     
#         Proof:
#             If we use only the train_train set for the model we would get a performance of NDCG = .6406
#                 Performance on training set: NDCG = .914
#             
#             If we use the train_train set to train a model and test it on the train_test set performance of NDCG = .364
#                 Performance on training set: NDCG = .915
#             
#             If we use only the train_test set for the model the performance is NDCG = .415
#                 Performance on training set: NDCG = .695
#                 
#         NOTE: These scores are all computed based on the same hyperparameters 
#                 and the same sizes of the training, validation and 
#                 test sets. Trainsize = 500 srch_ids, Valsize = 500 srch_ids, Testsize = 1000 srch_ids
