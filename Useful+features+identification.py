
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from collections import OrderedDict

from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score

id_set = list(OrderedDict.fromkeys(cleandata['id']))

#create a base for the dataframe (including the future mood variable)
persondf = cleandata[cleandata['id'] == 'AS14.02']
persondf['mood_f'] = persondf['mood].shift(1)


#find the correlations between the mood on a day and the variables i days ago
for i in range(1, 6):
    testdf =  pd.DataFrame(columns = persondf.columns)
    print('\t\tOFFSET:', i, 'day(s) ago\n')
    #shift the values for mood so that the variables of a day are followed by the mood of a day in the future
    for person in id_set:
        persondf = cleandata[cleandata['id'] == person]
        persondf['mood_f'] = persondf['mood'].shift(-i)
        testdf = testdf.append(persondf)
    testdf = testdf[testdf['mood_f'].notnull()]
    
    #test the correlation between mood and the other variables
    y = testdf['mood_f']
    x = testdf.drop('mood_f', 1)

    print('\tSIGNIFICANT CORRELATIONS:\n')
    sign_corr = list()
    for variable in x.columns[2:]:
        r, p = pearsonr(x[variable], y)
        if p < .05:
            sign_corr.append(variable)
            print(variable, '|| r:', r.round(3), 'p:', p.round(3))
    
    #perform a regression on mood with the significant correlations to find the number of days from the past that are useful
    x = x[sign_corr]
    lin_reg = LinearRegression()
    lin_reg.fit(x, y)
    
    print(\n\tExplained Variance:', explained_variance_score(y, lin_reg.predict(x)))
                              
    print('\n--------------------------------------------------------------------\n')

