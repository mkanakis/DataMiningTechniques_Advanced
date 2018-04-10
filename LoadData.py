# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 11:53:40 2018

@author: sabaa
"""

import pandas as pd
import numpy as np
# load data into dataframe
df = pd.read_csv('dataset_mood_smartphone.csv')
# add data part to dataframe
df['datepart'] = df.time.str[:10]
# pivot the data
pt = pd.pivot_table(df, values='value', index=['id', 'datepart'], columns='variable').reset_index()  # 1973 rows
# remove rows with no mood, valence, and arousal value--1268 rows
ptcl1=pt[np.isfinite(pt.mood)]  # 1268 rows

ptcl2 = ptcl1[np.isfinite(ptcl1['circumplex.valence'])]  # 1266 rows
ptcl3 = ptcl2[np.isfinite(ptcl2['circumplex.arousal'])]  # 1266 rows
# replace null with 0
cleandata = ptcl3.fillna(0)
del ptcl1, ptcl2, ptcl3, pt
