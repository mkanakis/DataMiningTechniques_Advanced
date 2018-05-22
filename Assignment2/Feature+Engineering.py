
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

df = pd.read_csv(r'C:\Users\MICK\Downloads\Data Mining VU data\Dataset_filled.csv')
df = df.drop('Unnamed: 0', 1)


# In[2]:


new_df = df.copy()
del df
print('data is filtered')


# In[3]:


# The mean of the property attributes that differ per search
ptprop = pd.pivot_table(new_df, values=['price_usd', 'prop_location_score2', 'prop_log_historical_price', 
                                        'invmean', 'ratemean', 'diffsmean'], 
                        index=['prop_id'], aggfunc='mean').reset_index()

ptprop.columns = ['prop_id', 'mean_diffsmean', 'mean_invmean', 'mean_price_usd', 'mean_prop_location_score2', 
                  'mean_prop_log_historical_price', 'mean_ratemean']
new_df = new_df.merge(ptprop, how='left', on='prop_id')

print('1')

# In[4]:


# The mean of the property attributes for the hotels shown in a search
ptid = pd.pivot_table(new_df, values=['prop_starrating',
       'prop_review_score', 'prop_location_score1',
       'prop_location_score2', 'prop_log_historical_price',
       'price_usd', 'invmean', 'ratemean', 'diffsmean'], index=['srch_id'], aggfunc='mean').reset_index()
ptid.columns = ['srch_id', 'srchid_mean_diffsmean', 'srchid_mean_invmean', 'srchid_mean_price_usd',
       'srchid_mean_prop_location_score1', 'srchid_mean_prop_location_score2',
       'srchid_mean_prop_log_historical_price', 'srchid_mean_prop_review_score', 
                'srchid_mean_prop_starrating', 'srchid_mean_ratemean']
new_df = new_df.merge(ptid, how='left', on='srch_id')
del ptid
print('2')

# In[5]:


# Add a incheck and checkout date
new_df.loc[:, 'date_time'] = pd.to_datetime(new_df['date_time'])

ptdate = pd.pivot_table(new_df, values=['srch_booking_window', 
                    'srch_length_of_stay'], index=['srch_id'], aggfunc='mean').reset_index()
ptdate.loc[:, 'srch_booking_window'] = pd.Series([pd.Timedelta(days=i) 
                    for i in ptdate['srch_booking_window']]) # convert it to a day-object
ptdate.loc[:, 'srch_length_of_stay'] = pd.Series([pd.Timedelta(days=i) 
                    for i in ptdate['srch_length_of_stay']]) # convert it to a day-object
ptdate.columns = ['srch_id', 'd_srch_booking_window', 'd_srch_length_of_stay']

new_df = new_df.merge(ptdate, how='left', on='srch_id')
new_df.loc[:, 'incheck_date'] = new_df['date_time'] + new_df['d_srch_booking_window']
new_df.loc[:, 'checkout_date'] = new_df['incheck_date'] + new_df['d_srch_length_of_stay']

new_df = new_df.drop(['d_srch_booking_window', 'd_srch_length_of_stay'], 1) # remove the attributes with the day-object
new_df.loc[:, 'incheck_date'] = pd.DatetimeIndex(new_df['incheck_date']).date # just take the date not the time
new_df.loc[:, 'checkout_date'] = pd.DatetimeIndex(new_df['checkout_date']).date # just take the date not the time

new_df.loc[:, 'booking_season'] = pd.DatetimeIndex(new_df['date_time']).quarter
new_df.loc[:, 'booking_month'] = pd.DatetimeIndex(new_df['date_time']).month
new_df.loc[:, 'booking_week'] = pd.DatetimeIndex(new_df['date_time']).week
new_df.loc[:, 'booking_weekday'] = pd.DatetimeIndex(new_df['date_time']).weekday
new_df.loc[:, 'booking_hour'] = pd.DatetimeIndex(new_df['date_time']).hour

new_df.loc[:, 'incheck_season'] = pd.DatetimeIndex(new_df['incheck_date']).quarter
new_df.loc[:, 'incheck_month'] = pd.DatetimeIndex(new_df['incheck_date']).month
new_df.loc[:, 'incheck_week'] = pd.DatetimeIndex(new_df['incheck_date']).week
new_df.loc[:, 'incheck_weekday'] = pd.DatetimeIndex(new_df['incheck_date']).weekday

new_df.loc[:, 'checkout_season'] = pd.DatetimeIndex(new_df['checkout_date']).quarter
new_df.loc[:, 'checkout_month'] = pd.DatetimeIndex(new_df['checkout_date']).month
new_df.loc[:, 'checkout_week'] = pd.DatetimeIndex(new_df['checkout_date']).week
new_df.loc[:, 'checkout_weekday'] = pd.DatetimeIndex(new_df['checkout_date']).weekday

new_df = new_df.drop(['incheck_date', 'checkout_date', 'date_time'], 1)
del ptdate
print('3')

# In[6]:


# Create difference scores between the means of all the results in a search and the specific result
    # and the difference between the specific results and the mean of the property in other searches
for column in ['prop_starrating',
       'prop_review_score', 'prop_location_score1',
       'prop_location_score2', 'prop_log_historical_price',
       'price_usd', 'invmean', 'ratemean', 'diffsmean']:
    
    srch_id = 'srchid_mean_' + column
    true_var_id = 'true_diff_w_srchresults_' + column
    abs_var_id = 'abs_diff_w_srch_results_' + column
    per_var_id = 'per_diff_w_srch_results_' + column
    
    if column in ['price_usd','prop_location_score2', 'prop_log_historical_price', 'invmean', 'ratemean', 'diffsmean']:
        mean_prop = 'mean_' + column
        true_var_mean = 'true_diff_w_mean_' + column
        abs_var_mean = 'abs_diff_w_mean' + column
        per_var_mean = 'per_diff_w_mean'+ column
        
        new_df.loc[:, true_var_mean] = new_df[column] - new_df[mean_prop]
        new_df.loc[:, abs_var_mean] = abs(new_df[column] - new_df[mean_prop])
        new_df.loc[:, per_var_mean] = (100*(new_df[column] - new_df[mean_prop]))/ new_df[mean_prop]
        
    
    new_df.loc[:, true_var_id] = new_df[column] - new_df[srch_id]
    new_df.loc[:, abs_var_id] = abs(new_df[column] - new_df[srch_id])
    new_df.loc[:, per_var_id] = (100*(new_df[column] - new_df[srch_id])) / new_df[srch_id]
    
    
    new_df = new_df.drop([srch_id], 1)
    
print('difference scores is done')
del srch_id, true_var_id, abs_var_id, per_var_id, mean_prop, true_var_mean, abs_var_mean, per_var_mean


# In[7]:


new_df.loc[:, 'true_diff_hist_star'] = new_df['prop_starrating'] - new_df['visitor_hist_starrating']
new_df.loc[:, 'true_diff_hist_review'] = new_df['prop_review_score'] - new_df['visitor_hist_starrating']
new_df.loc[:, 'true_diff_hist_price'] = new_df['price_usd'] - new_df['visitor_hist_adr_usd']

new_df.loc[:, 'abs_diff_hist_star'] = abs(new_df['prop_starrating'] - new_df['visitor_hist_starrating'])
new_df.loc[:, 'abs_diff_hist_review'] = abs(new_df['prop_review_score'] - new_df['visitor_hist_starrating'])
new_df.loc[:, 'abs_diff_hist_price'] = abs(new_df['price_usd'] - new_df['visitor_hist_adr_usd'])

new_df.loc[:, 'per_diff_hist_star'] = (100*(new_df['prop_starrating'] - new_df['visitor_hist_starrating'])) / new_df['visitor_hist_starrating']
new_df.loc[:, 'per_diff_hist_review'] = (100*(new_df['prop_review_score'] - new_df['visitor_hist_starrating'])) / new_df['visitor_hist_starrating']
new_df.loc[:, 'per_diff_hist_price'] = (100*(new_df['price_usd'] - new_df['visitor_hist_adr_usd'])) / new_df['visitor_hist_adr_usd']

new_df.loc[:, 'relevance'] = new_df['click_bool'] + 4*new_df['booking_bool']
new_df = new_df.drop('gross_bookings_usd', 1)


# In[ ]:


from sklearn.model_selection import train_test_split as tts
srch_ids = list(set(new_df['srch_id']))

x_train, x_test, y_train, y_test = tts(srch_ids, srch_ids, test_size = .5)

del y_train, y_test

train = new_df[new_df['srch_id'].isin(x_train)]
test = new_df[new_df['srch_id'].isin(x_test)]


# In[ ]:


pt = pd.pivot_table(train, values=['click_bool', 'booking_bool'], index=['prop_id'], aggfunc='mean').reset_index()
pt.columns = ['prop_id', 'mean_booking', 'mean_click']

train = train.merge(pt, how='left', on='prop_id')
test = test.merge(pt, how='left', on='prop_id')

del pt

pt = pd.pivot_table(train[train['random_bool'] == 0], values='position', index=['prop_id'], aggfunc='mean').reset_index()
pt.columns = ['prop_id', 'mean_position']

train = train.merge(pt, how='left', on='prop_id')
test = test.merge(pt, how='left', on='prop_id')

del pt


# In[ ]:


test.loc[:, 'mean_booking'] = test['mean_booking'].fillna(test['mean_booking'].mean())
test.loc[:, 'mean_click'] = test['mean_click'].fillna(test['mean_click'].mean())
test.loc[:, 'mean_position'] = test['mean_position'].fillna(test['mean_position'].mean())
train.loc[:, 'mean_position'] = train['mean_position'].fillna(train['mean_position'].mean())


# In[ ]:


# The mean of the property attributes for the hotels shown in a search
ptid = pd.pivot_table(test, values=['mean_click',
       'mean_booking', 'mean_position'], index=['srch_id'], aggfunc='mean').reset_index()
ptid2 = pd.pivot_table(train, values=['mean_click',
       'mean_booking', 'mean_position'], index=['srch_id'], aggfunc='mean').reset_index()


# In[ ]:


ptid.columns = ['srch_id', 'srchid_mean_mean_booking', 'srchid_mean_mean_click', 'srchid_mean_mean_position']
ptid2.columns = ['srch_id', 'srchid_mean_mean_booking', 'srchid_mean_mean_click', 'srchid_mean_mean_position']

test = test.merge(ptid, how='left', on='srch_id')
train = train.merge(ptid2, how='left', on = 'srch_id')
del ptid, ptid2


# In[ ]:


for column in ['mean_booking', 'mean_click', 'mean_position']:
    srch_id = 'srchid_mean_' + column
    true_var_id = 'true_diff_w_srchresults_' + column
    abs_var_id = 'abs_diff_w_srch_results_' + column
    per_var_id = 'per_diff_w_srch_results_' + column
    
    test.loc[:, true_var_id] = test[column] - test[srch_id]
    test.loc[:, abs_var_id] = abs(test[column] - test[srch_id])
    test.loc[:, per_var_id] = (100*(test[column] - test[srch_id])) / test[srch_id]
    
    
    test = test.drop([srch_id], 1)
    
for column in ['mean_booking', 'mean_click', 'mean_position']:
    srch_id = 'srchid_mean_' + column
    true_var_id = 'true_diff_w_srchresults_' + column
    abs_var_id = 'abs_diff_w_srch_results_' + column
    per_var_id = 'per_diff_w_srch_results_' + column
    
    train.loc[:, true_var_id] = train[column] - train[srch_id]
    train.loc[:, abs_var_id] = abs(train[column] - train[srch_id])
    train.loc[:, per_var_id] = (100*(train[column] - train[srch_id])) / train[srch_id]
    
    
    train = train.drop([srch_id], 1)
    
del srch_id, true_var_id, abs_var_id, per_var_id


# In[ ]:


# Add features for the search features. Compare the srch features with the weighted srch features for the properties
    # when they were booked.
WEIGHT_OF_RELEVANCE = 50    
train.loc[:,'relevance2'] = (train['relevance']*WEIGHT_OF_RELEVANCE) + 1

for column in ['srch_length_of_stay', 'srch_booking_window',
       'srch_adults_count', 'srch_children_count', 'srch_room_count',
       'srch_saturday_night_bool']:
    train.loc[:, 'weighted_' + column] = train[column] * train['relevance2']

pt_srch = pd.pivot_table(train, values=['weighted_srch_length_of_stay',
       'weighted_srch_booking_window', 'weighted_srch_adults_count',
       'weighted_srch_children_count', 'weighted_srch_room_count',
       'weighted_srch_saturday_night_bool', 'relevance2'], index=['prop_id'], aggfunc='sum').reset_index()
for column in ['weighted_srch_length_of_stay', 'weighted_srch_booking_window', 'weighted_srch_adults_count',
       'weighted_srch_children_count', 'weighted_srch_room_count',
        'weighted_srch_saturday_night_bool']:
    pt_srch.loc[:, column] = pt_srch[column]/pt_srch['relevance2']
pt_srch = pt_srch.drop('relevance2', 1)

train = train.drop(['weighted_srch_length_of_stay',
       'weighted_srch_booking_window', 'weighted_srch_adults_count',
       'weighted_srch_children_count', 'weighted_srch_room_count',
       'weighted_srch_saturday_night_bool', 'relevance2'],1)


# In[ ]:


train = train.merge(pt_srch, how='left', on='prop_id')
test = test.merge(pt_srch, how='left', on='prop_id')

for column in ['weighted_srch_length_of_stay',
       'weighted_srch_booking_window', 'weighted_srch_adults_count',
       'weighted_srch_children_count', 'weighted_srch_room_count',
       'weighted_srch_saturday_night_bool']:
    test.loc[:, column] = test[column].fillna(pt_srch[column].mean())

for column in ['srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
       'srch_children_count', 'srch_room_count',
        'srch_saturday_night_bool']:
    weighted = 'weighted_' + column
    true_new_var = 'true_diff_prop_' + column
    abs_new_var = 'abs_diff_prop_' + column
    per_new_var = 'per_diff_prop_' + column    
    
    train.loc[:, true_new_var] = train[column] - train[weighted]
    test.loc[:, true_new_var] = test[column] - test[weighted]
    
    train.loc[:, abs_new_var] = abs(train[column] - train[weighted])
    test.loc[:, abs_new_var] = abs(test[column] - test[weighted])
    
    train.loc[:, per_new_var] = (100*(train[column] - train[weighted])) / train[weighted]
    test.loc[:, per_new_var] = (100*(test[column] - test[weighted])) / test[weighted]

del pt_srch, column, weighted, true_new_var, abs_new_var, per_new_var, WEIGHT_OF_RELEVANCE
print('Comparison between srch features and the weighted srch features of the property when booked is done')


# In[ ]:


train.to_csv('F:\Dataset_train.csv')
test.to_csv('F:\Dataset_test.csv')

