"""
Created on Tue Apr 10 11:53:40 2018

@author: sabaa, mick
"""

import pandas as pd
import numpy as np


# Returns data after cleaning them from csv-formatted file
def retdata():
    # load data into dataframe
    df = pd.read_csv('dataset_mood_smartphone.csv')
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

    # replace null with 0
    cleandata = ptcl3.fillna(0)
    del ptcl1, ptcl2, ptcl3, pt, pt1, pt2, df, df2, means
    return cleandata
