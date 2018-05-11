
# coding: utf-8

# In[21]:


PERCENT_TOP = .009 # $513
PERCENT_BOT = .001 # $23.8

import numpy as np
import pandas as pd

df = pd.read_csv('Dataset_Complete.csv')
df = df.drop('Unnamed: 0', 1)
print('data is loaded')


# remove the srch_ids with weird price_usd values
pt = pd.pivot_table(df, values='price_usd', index=['srch_id'], aggfunc='mean').reset_index()

max_v = (pt['price_usd'].quantile(q=1 - PERCENT_TOP))
min_v = (pt['price_usd'].quantile(q=PERCENT_BOT))
print(f'min price: {min_v} ({PERCENT_BOT}) \tmax price: {max_v} ({PERCENT_TOP})')

srch_ids_to_keep = set(pt[(pt['price_usd'] < max_v) & (pt['price_usd'] > min_v)]['srch_id'])

new_df = df[df['srch_id'].isin(srch_ids_to_keep)].reset_index()
del df, pt, max_v, min_v, srch_ids_to_keep
print('data is filtered')


# The mean of the property attributes that differ per search
ptprop = pd.pivot_table(new_df, values=['price_usd', 
        'prop_location_score2', 'prop_log_historical_price', 'position', 
        'booking_bool', 'click_bool'], index=['prop_id'], aggfunc='mean').reset_index()
ptprop.columns = ['prop_id', 'mean_booking', 'mean_click', 'mean_position', 'mean_price_usd', 
                  'mean_prop_location_score2', 'mean_prop_log_historical_price']
new_df = new_df.merge(ptprop, how='left', on='prop_id')
del ptprop
print('mean of each property in other searches is done')


# The mean of the property attributes for the hotels shown in a search
ptid = pd.pivot_table(new_df, values=['prop_starrating',
       'prop_review_score', 'prop_location_score1',
       'prop_location_score2', 'prop_log_historical_price',
       'price_usd'], index=['srch_id'], aggfunc='mean').reset_index()
ptid.columns = ['srch_id', 'srchid_mean_price_usd',
       'srchid_mean_prop_location_score1', 'srchid_mean_prop_location_score2',
       'srchid_mean_prop_log_historical_price', 'srchid_mean_prop_review_score', 'srchid_mean_prop_starrating']
new_df = new_df.merge(ptid, how='left', on='srch_id')
del ptid
print('mean of the property attributes in same search is done')


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
print('incheck and checkout date is done')


# Create difference scores between the means of all the results in a search and the specific result
    # and the difference between the specific results and the mean of the property in other searches
for column in ['prop_starrating',
       'prop_review_score', 'prop_location_score1',
       'prop_location_score2', 'prop_log_historical_price',
       'price_usd']:
    
    srch_id = 'srchid_mean_' + column
    new_var_id = 'diff_w_results_' + column
    
    if column in ['price_usd','prop_location_score2', 'prop_log_historical_price']:
        mean_prop = 'mean_' + column
        new_var_mean = 'diff_w_mean_' + column
        new_df.loc[:, new_var_mean] = new_df[column] - new_df[mean_prop]
    
    new_df.loc[:, new_var_id] = new_df[column] - new_df[srch_id]
    new_df = new_df.drop([srch_id], 1)
print('difference scores is done')
del srch_id, new_var_id, column, mean_prop, new_var_mean

# Add a relevance feature and remove the click/booking bool
new_df.loc[:, 'diff_hist_star'] = new_df['prop_starrating'] - new_df['visitor_hist_starrating']
new_df.loc[:, 'diff_hist_price'] = new_df['price_usd'] - new_df['visitor_hist_adr_usd']
new_df.loc[:, 'relevance'] = new_df['click_bool'] + 4*new_df['booking_bool']
new_df = new_df.drop(['click_bool', 'booking_bool'], 1)

