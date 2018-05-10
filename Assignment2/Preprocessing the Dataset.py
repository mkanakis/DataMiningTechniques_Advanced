
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np

df = pd.read_csv('training_set_VU_DM_2014.csv')
df = df[:5000]
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


#filling in the NA values
df = df[['srch_id', 'date_time', 'site_id', 'visitor_location_country_id',
       'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id',
       'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool',
       'prop_location_score1', 'prop_location_score2',
       'prop_log_historical_price', 'position', 'price_usd', 'promotion_flag',
       'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window',
       'srch_adults_count', 'srch_children_count', 'srch_room_count',
       'srch_saturday_night_bool', 'srch_query_affinity_score',
       'orig_destination_distance', 'random_bool', 'click_bool', 
       'gross_bookings_usd', 'booking_bool']]

df.loc[:, 'prop_review_score'] = df['prop_review_score'].fillna(0)
df.loc[:, 'prop_location_score2'] = df['prop_location_score2'].fillna(0)
df = pd.concat([df, dfcompetitors], axis=1, join_axes=[df.index])
del dfcompetitors

