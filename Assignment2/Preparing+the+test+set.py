
# coding: utf-8

# In[1]:


#Extract the information from the training set that we want to impute in the test set. e.a. mean_booking/click/position

import pandas as pd
import numpy as np


TRAINING_SET_LOCATION = r'F:\Vu Datamining files\training_set_VU_DM_2014.csv'
TEST_SET_LOCATION = r'F:\Vu Datamining files\Vu_test_set_with_scores.csv'
MODEL_LOCATION = r'C:\Users\MICK\Downloads\Data Mining VU data\model515.sav'
TEST_COMPLETE_SAVE_LOCATION = r'F:\Vu Datamining files\Vu_test_set_with_scores_and_features.csv'
PREDICTIONS_SAVE_LOCATION = r'F:\Vu_test.csv'

df = pd.read_csv(TRAINING_SET_LOCATION)
df = df[['srch_id', 'prop_id', 'position', 'click_bool', 'random_bool', 'booking_bool', 'srch_length_of_stay', 'srch_booking_window',
       'srch_adults_count', 'srch_children_count', 'srch_room_count',
       'srch_saturday_night_bool']]
print('data is loaded')

ptclickbook = pd.pivot_table(df, values=['click_bool', 'booking_bool'], index=['prop_id'], aggfunc='mean').reset_index()
ptclickbook.columns = ['prop_id', 'mean_booking', 'mean_click']

ptposition = pd.pivot_table(df[df['random_bool'] == 0], values='position', index=['prop_id'], aggfunc='mean').reset_index()
ptposition.columns = ['prop_id', 'mean_position']


# Add features for the search features. Compare the srch features with the weighted srch features for the properties
    # when they were booked.
df.loc[:, 'relevance'] = df['click_bool'] + 4*df['booking_bool']

WEIGHT_OF_RELEVANCE = 50    
df.loc[:,'relevance2'] = (df['relevance']*WEIGHT_OF_RELEVANCE) + 1

for column in ['srch_length_of_stay', 'srch_booking_window',
       'srch_adults_count', 'srch_children_count', 'srch_room_count',
       'srch_saturday_night_bool']:
    df.loc[:, 'weighted_' + column] = df[column] * df['relevance2']

pt_srch = pd.pivot_table(df, values=['weighted_srch_length_of_stay',
       'weighted_srch_booking_window', 'weighted_srch_adults_count',
       'weighted_srch_children_count', 'weighted_srch_room_count',
       'weighted_srch_saturday_night_bool', 'relevance2'], index=['prop_id'], aggfunc='sum').reset_index()

for column in ['weighted_srch_length_of_stay', 'weighted_srch_booking_window', 'weighted_srch_adults_count',
       'weighted_srch_children_count', 'weighted_srch_room_count',
        'weighted_srch_saturday_night_bool']:
    pt_srch.loc[:, column] = pt_srch[column]/pt_srch['relevance2']
pt_srch = pt_srch.drop('relevance2', 1)

del df, column

# In[2]:


df = pd.read_csv(TEST_SET_LOCATION)


print('data is loaded')


#Visitor history starrating
ptstar1 = pd.pivot_table(df, values='visitor_hist_starrating', index=['visitor_location_country_id', 'srch_destination_id'], aggfunc='mean').dropna().reset_index()
new_df = pd.merge(df.drop('visitor_hist_starrating', 1), ptstar1, how='left', on=['visitor_location_country_id', 'srch_destination_id'])
df.loc[:, 'visitor_hist_starrating'] = df['visitor_hist_starrating'].fillna(new_df['visitor_hist_starrating'])
del ptstar1, new_df

ptstar2 = pd.pivot_table(df, values='visitor_hist_starrating', index=['visitor_location_country_id', 'prop_country_id'], aggfunc='mean').dropna().reset_index()
new_df = pd.merge(df.drop('visitor_hist_starrating', 1), ptstar2, how='left', on=['visitor_location_country_id', 'prop_country_id'])
df.loc[:, 'visitor_hist_starrating'] = df['visitor_hist_starrating'].fillna(new_df['visitor_hist_starrating'])
del ptstar2, new_df

ptstar3 = pd.pivot_table(df, values='visitor_hist_starrating', index=['visitor_location_country_id'], aggfunc = 'mean').dropna().reset_index()
new_df = pd.merge(df.drop('visitor_hist_starrating', 1), ptstar3, how='left', on=['visitor_location_country_id'])
df.loc[:, 'visitor_hist_starrating'] = df['visitor_hist_starrating'].fillna(new_df['visitor_hist_starrating'])
del ptstar3, new_df

df.loc[:, 'visitor_hist_starrating'] = df['visitor_hist_starrating'].fillna(df['visitor_hist_starrating'].mean())
print('visitor_hist_starrating is done')


#Visitor history price
ptprice1 = pd.pivot_table(df, values='visitor_hist_adr_usd', index=['visitor_location_country_id', 'srch_destination_id'], aggfunc='mean').dropna().reset_index()
new_df = pd.merge(df.drop('visitor_hist_adr_usd', 1), ptprice1, how='left', on=['visitor_location_country_id', 'srch_destination_id'])
df.loc[:, 'visitor_hist_adr_usd'] = df['visitor_hist_adr_usd'].fillna(new_df['visitor_hist_adr_usd'])
del ptprice1, new_df

ptprice2 = pd.pivot_table(df, values='visitor_hist_adr_usd', index=['visitor_location_country_id', 'prop_country_id'], aggfunc='mean').dropna().reset_index()
new_df = pd.merge(df.drop('visitor_hist_adr_usd', 1), ptprice2, how='left', on=['visitor_location_country_id', 'prop_country_id'])
df.loc[:, 'visitor_hist_adr_usd'] = df['visitor_hist_adr_usd'].fillna(new_df['visitor_hist_adr_usd'])
del ptprice2, new_df

ptprice3 = pd.pivot_table(df, values='visitor_hist_adr_usd', index=['visitor_location_country_id'], aggfunc = 'mean').dropna().reset_index()
new_df = pd.merge(df.drop('visitor_hist_adr_usd', 1), ptprice3, how='left', on=['visitor_location_country_id'])
df.loc[:, 'visitor_hist_adr_usd'] = df['visitor_hist_adr_usd'].fillna(new_df['visitor_hist_adr_usd'])
del ptprice3, new_df

df.loc[:, 'visitor_hist_adr_usd'] = df['visitor_hist_adr_usd'].fillna(df['visitor_hist_adr_usd'].mean())
print('visitor_hist_adr_usd is done')


#Srch query affinity Score
ptquery1 = pd.pivot_table(df, values='srch_query_affinity_score', index=['prop_id'], aggfunc='mean').dropna().reset_index()
new_df = pd.merge(df.drop('srch_query_affinity_score', 1), ptquery1, how='left', on=['prop_id'])
df.loc[:, 'srch_query_affinity_score'] = df['srch_query_affinity_score'].fillna(new_df['srch_query_affinity_score'])
del ptquery1, new_df

ptquery2 = pd.pivot_table(df, values='srch_query_affinity_score', index=['site_id', 'srch_destination_id'], aggfunc='mean').dropna().reset_index()
new_df = pd.merge(df.drop('srch_query_affinity_score', 1), ptquery2, how='left', on=['site_id', 'srch_destination_id'])
df.loc[:, 'srch_query_affinity_score'] = df['srch_query_affinity_score'].fillna(new_df['srch_query_affinity_score'])
del ptquery2, new_df

ptquery3 = pd.pivot_table(df, values='srch_query_affinity_score', index=['site_id', 'prop_country_id'], aggfunc = 'mean').dropna().reset_index()
new_df = pd.merge(df.drop('srch_query_affinity_score', 1), ptquery3, how='left', on=['site_id', 'prop_country_id'])
df.loc[:, 'srch_query_affinity_score'] = df['srch_query_affinity_score'].fillna(new_df['srch_query_affinity_score'])
del ptquery3, new_df

df.loc[:, 'srch_query_affinity_score'] = df['srch_query_affinity_score'].fillna(df['srch_query_affinity_score'].mean())
print('srch_query_affinity_score is done')


#Origin_destination_distance
ptdist1 = pd.pivot_table(df, values='orig_destination_distance', index=['visitor_location_country_id', 'srch_destination_id'], aggfunc='mean').dropna().reset_index()
new_df = pd.merge(df.drop('orig_destination_distance', 1), ptdist1, how='left', on=['visitor_location_country_id', 'srch_destination_id'])
df.loc[:, 'orig_destination_distance'] = df['orig_destination_distance'].fillna(new_df['orig_destination_distance'])
del ptdist1, new_df

ptdist2 = pd.pivot_table(df, values='orig_destination_distance', index=['visitor_location_country_id', 'prop_country_id'], aggfunc='mean').dropna().reset_index()
new_df = pd.merge(df.drop('orig_destination_distance', 1), ptdist2, how='left', on=['visitor_location_country_id', 'prop_country_id'])
df.loc[:, 'orig_destination_distance'] = df['orig_destination_distance'].fillna(new_df['orig_destination_distance'])
del ptdist2, new_df

ptdist3 = pd.pivot_table(df, values='orig_destination_distance', index=['visitor_location_country_id'], aggfunc = 'mean').dropna().reset_index()
new_df = pd.merge(df.drop('orig_destination_distance', 1), ptdist3, how='left', on=['visitor_location_country_id'])
df.loc[:, 'orig_destination_distance'] = df['orig_destination_distance'].fillna(new_df['orig_destination_distance'])
del ptdist3, new_df

df.loc[:, 'orig_destination_distance'] = df['orig_destination_distance'].fillna(df['orig_destination_distance'].mean())
print('orig_destination_distance is done')


#Competitor variables
rates = ["comp1_rate", "comp2_rate", "comp3_rate", "comp4_rate",
         "comp5_rate", "comp6_rate", "comp7_rate", "comp8_rate"]
invs = ['comp1_inv', 'comp2_inv', 'comp3_inv', 'comp4_inv',
       'comp5_inv', 'comp6_inv', 'comp7_inv', 'comp8_inv']
diffs = ['comp1_rate_percent_diff', 'comp2_rate_percent_diff', 'comp3_rate_percent_diff',
        'comp4_rate_percent_diff', 'comp5_rate_percent_diff', 'comp6_rate_percent_diff',
        'comp7_rate_percent_diff', 'comp8_rate_percent_diff']

dfrates = df[rates]
dfinvs = df[invs]
dfdiffs = df[diffs]

ratemean = []
invsmean = []
diffsmean =[]
j = 0
for i in range(len(dfrates)):
    rates = []
    for value in dfrates.values[i]:
        if value in [-1, 0, 1]:
            rates.append(value)    
    if len(rates) == 0:
        ratemean.append(np.nan)
    else:
        ratemean.append(sum(rates)/len(rates))
        
    invs = []
    for value in dfinvs.values[i]:
        if value in [-1, 0, 1]:
            invs.append(value)
    
    if len(invs) == 0:
        invsmean.append(np.nan)
    else:
        invsmean.append(sum(invs)/len(invs))
        
    diffs = []
    for value in dfdiffs.values[i]:
        if (value < 500):
            diffs.append(value)
    
    if len(diffs) == 0:
        diffsmean.append(np.nan)
    else:
        diffsmean.append(sum(diffs)/len(diffs))
    
    if (i != 0) and (i%(len(dfrates)/10) == 0):
        j += 1
        print(j, '%', end = ' ')

invsmean = np.asarray(invsmean)
ratemean = np.asarray(ratemean)
diffsmean = np.asarray(diffsmean)

dfcompetitors = pd.DataFrame()
dfcompetitors.loc[:, 'invmean'] = invsmean
dfcompetitors.loc[:, 'ratemean'] = ratemean
dfcompetitors.loc[:, 'diffsmean'] = diffsmean

dfcompetitors.loc[:, 'ratemean'] = dfcompetitors['ratemean'].fillna(dfcompetitors['ratemean'].mean())
dfcompetitors.loc[:, 'invmean'] = dfcompetitors['invmean'].fillna(dfcompetitors['ratemean'].min())
dfcompetitors.loc[:, 'diffsmean'] = dfcompetitors['diffsmean'].fillna(dfcompetitors['diffsmean'].min())
print('competitor data is done')

del dfdiffs, dfinvs, dfrates, diffs, diffsmean, i, invs, invsmean, j, ratemean, rates, value

#filling in the NA values
df = df[['srch_id', 'date_time', 'click_bool', 'booking_bool', 'site_id', 'visitor_location_country_id',
       'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id',
       'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool',
       'prop_location_score1', 'prop_location_score2',
       'prop_log_historical_price', 'price_usd', 'promotion_flag',
       'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window',
       'srch_adults_count', 'srch_children_count', 'srch_room_count',
       'srch_saturday_night_bool', 'srch_query_affinity_score',
       'orig_destination_distance', 'random_bool']]

df.loc[:, 'prop_review_score'] = df['prop_review_score'].fillna(0)
df.loc[:, 'prop_location_score2'] = df['prop_location_score2'].fillna(0)
df = pd.concat([df, dfcompetitors], axis=1, join_axes=[df.index])
del dfcompetitors

# In[3]:


new_df = df.copy()
del df

# The mean of the property attributes that differ per search
ptprop = pd.pivot_table(new_df, values=['price_usd', 'prop_location_score2', 'prop_log_historical_price', 
                                        'invmean', 'ratemean', 'diffsmean'], 
                        index=['prop_id'], aggfunc='mean').reset_index()

ptprop.columns = ['prop_id', 'mean_diffsmean', 'mean_invmean', 'mean_price_usd', 'mean_prop_location_score2', 
                  'mean_prop_log_historical_price', 'mean_ratemean']
new_df = new_df.merge(ptprop, how='left', on='prop_id')
del ptprop


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
del srch_id, true_var_id, abs_var_id, per_var_id, mean_prop, true_var_mean, abs_var_mean, per_var_mean, column

new_df.loc[:, 'true_diff_hist_star'] = new_df['prop_starrating'] - new_df['visitor_hist_starrating']
new_df.loc[:, 'true_diff_hist_review'] = new_df['prop_review_score'] - new_df['visitor_hist_starrating']
new_df.loc[:, 'true_diff_hist_price'] = new_df['price_usd'] - new_df['visitor_hist_adr_usd']

new_df.loc[:, 'abs_diff_hist_star'] = abs(new_df['prop_starrating'] - new_df['visitor_hist_starrating'])
new_df.loc[:, 'abs_diff_hist_review'] = abs(new_df['prop_review_score'] - new_df['visitor_hist_starrating'])
new_df.loc[:, 'abs_diff_hist_price'] = abs(new_df['price_usd'] - new_df['visitor_hist_adr_usd'])

new_df.loc[:, 'per_diff_hist_star'] = (100*(new_df['prop_starrating'] - new_df['visitor_hist_starrating'])) / new_df['visitor_hist_starrating']
new_df.loc[:, 'per_diff_hist_review'] = (100*(new_df['prop_review_score'] - new_df['visitor_hist_starrating'])) / new_df['visitor_hist_starrating']
new_df.loc[:, 'per_diff_hist_price'] = (100*(new_df['price_usd'] - new_df['visitor_hist_adr_usd'])) / new_df['visitor_hist_adr_usd']



#Adding the mean_click/booking/position for each property based on the training set
new_df = new_df.merge(ptclickbook, how='left', on='prop_id')
new_df = new_df.merge(ptposition, how='left', on='prop_id')
del ptclickbook, ptposition

new_df.loc[:, 'mean_click'] = new_df['mean_click'].fillna(new_df['mean_click'].mean())
new_df.loc[:, 'mean_booking'] = new_df['mean_booking'].fillna(new_df['mean_booking'].mean())
new_df.loc[:, 'mean_position'] = new_df['mean_position'].fillna(new_df['mean_position'].mean())

ptid = pd.pivot_table(new_df, values=['mean_click',
       'mean_booking', 'mean_position'], index=['srch_id'], aggfunc='mean').reset_index()
ptid.columns = ['srch_id', 'srchid_mean_mean_booking', 'srchid_mean_mean_click', 'srchid_mean_mean_position']
new_df = new_df.merge(ptid, how='left', on='srch_id')
del ptid

for column in ['mean_booking', 'mean_click', 'mean_position']:
    srch_id = 'srchid_mean_' + column
    true_var_id = 'true_diff_w_srchresults_' + column
    abs_var_id = 'abs_diff_w_srch_results_' + column
    per_var_id = 'per_diff_w_srch_results_' + column
    
    new_df.loc[:, true_var_id] = new_df[column] - new_df[srch_id]
    new_df.loc[:, abs_var_id] = abs(new_df[column] - new_df[srch_id])
    new_df.loc[:, per_var_id] = (100*(new_df[column] - new_df[srch_id])) / new_df[srch_id]
    
    new_df = new_df.drop([srch_id], 1)
    
del srch_id, true_var_id, abs_var_id, per_var_id, column

#Adding the weighted srch attributes for each property based on the training set

new_df = new_df.merge(pt_srch, how='left', on='prop_id')

for column in ['weighted_srch_length_of_stay',
       'weighted_srch_booking_window', 'weighted_srch_adults_count',
       'weighted_srch_children_count', 'weighted_srch_room_count',
       'weighted_srch_saturday_night_bool']:
    new_df.loc[:, column] = new_df[column].fillna(pt_srch[column].mean())
    
for column in ['srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
       'srch_children_count', 'srch_room_count',
        'srch_saturday_night_bool']:
    
    weighted = 'weighted_' + column
    true_new_var = 'true_diff_prop_' + column
    abs_new_var = 'abs_diff_prop_' + column
    per_new_var = 'per_diff_prop_' + column    
    
    new_df.loc[:, true_new_var] = new_df[column] - new_df[weighted] 
    new_df.loc[:, abs_new_var] = abs(new_df[column] - new_df[weighted])
    new_df.loc[:, per_new_var] = (100*(new_df[column] - new_df[weighted])) / new_df[weighted]
    
del pt_srch, column, weighted, true_new_var, abs_new_var, per_new_var, WEIGHT_OF_RELEVANCE

new_df = new_df.replace([np.inf, -np.inf], np.nan)
new_df = new_df.fillna(0)

new_df.loc[:, 'relevance'] = new_df['click_bool'] + 4*new_df['booking_bool']
new_df.to_csv(TEST_COMPLETE_SAVE_LOCATION)