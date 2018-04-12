"""
Created on Tue Apr 10 11:53:40 2018
@author: Saba, Mick
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing


#normalizes the whole dataset
def normalize(df):
    
    result = df.copy()
    for feature_name in df.columns:
        if feature_name not in ['mood','circumplex.valence','circumplex.arousal','id','datepart']:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result
#normalizes the dataset per user
def normalizeperuser(df):
    
    dict_of_users = {k: v for k, v in df.groupby('id')}
    result =  pd.DataFrame(columns = df.columns)
    for udf in dict_of_users:
        result = result.append(normalize(dict_of_users[udf]))
    return result
# Returns data after cleaning and normalizing them(both over the whole dataset and per user) from csv-formatted file

def retdata():
    # load data into dataframe
    df = pd.read_csv('ds.csv')
    df = df.drop('Unnamed: 0', 1)

    # add data part to dataframe
    df['datepart'] = pd.DatetimeIndex(pd.to_datetime(df['time'])).date

    # split dataframe up into the variable where we want the sum and where we want the mean
    means = ['mood', 'circumplex.valence', 'circumplex.arousal']
    df2 = df[df['variable'].isin(means)]
    df = df[~df['variable'].isin(means)]
    
    # create 2 different dataframes with different aggfunc and merge them
    pt1 = pd.pivot_table(df, values='value', index=['id', 'datepart'], aggfunc='sum',
                         columns='variable').reset_index()  # 1973 rows
    pt2 = pd.pivot_table(df2, values='value', index=['id', 'datepart'], aggfunc='mean',
                         columns='variable').reset_index()  # 1973 rows
    pt = pd.merge(pt1, pt2, how='left', left_on=['id', 'datepart'], right_on=['id', 'datepart'])

    # remove rows with no mood, valence, and arousal value--1268 rows
    ptcl1 = pt[np.isfinite(pt.mood)]  # 1268 rows
    ptcl2 = ptcl1[np.isfinite(ptcl1['circumplex.valence'])]  # 1266 rows
    ptcl3 = ptcl2[np.isfinite(ptcl2['circumplex.arousal'])]  # 1266 rows
    #add previous day's mood
    for feature_name in ptcl3.columns:
        if feature_name not in ['id','datepart']:
            ptcl3[str(feature_name)+'PrevDay'] = ptcl3[feature_name].shift(1)
            ptcl3[str(feature_name)+'MeanPrevDays'] = ptcl3[str(feature_name)+'PrevDay'].rolling(5).mean()
            ptcl3[str(feature_name)+'Gradient'] = np.gradient(ptcl3[str(feature_name)+'PrevDay'].rolling(center=False,window=5).mean())

    ptcl3['moodprevday'] = ptcl3['mood'].shift(1)
    #add mean of the last n days mood mean
    ptcl3['moodmeanprevdays'] = ptcl3['moodprevday'].rolling(5).mean()
    ptcl3['Gradient'] = np.gradient(ptcl3['moodprevday'].rolling(center=False,window=5).mean())
    # replace null with 0
    cleandata = ptcl3.fillna(0)
    #clean up
    del ptcl1, ptcl2, ptcl3, pt, pt1, pt2, df, df2, means
    #get normalized datasets
    normalizedwholeds = normalize(cleandata)
    normalizedperuser = normalizeperuser(cleandata)    
    
    return normalizedwholeds,normalizedperuser


dsNormalizedWhole, dsNormalizedPerUser = retdata()
