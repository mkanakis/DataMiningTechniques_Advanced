
# coding: utf-8

# In[ ]:


#the preferred order for the DataFrame (just for inspection)
cols = ['id',
 'datepart',
 'mood_',
 'weekday',
 'activityGradient',
 'activityMeanPrevDays',
 'activityPrevDay',
 'appCat.builtinGradient',
 'appCat.builtinMeanPrevDays',
 'appCat.builtinPrevDay',
 'appCat.communicationGradient',
 'appCat.communicationMeanPrevDays',
 'appCat.communicationPrevDay',
 'appCat.entertainmentGradient',
 'appCat.entertainmentMeanPrevDays',
 'appCat.entertainmentPrevDay',
 'appCat.financeGradient',
 'appCat.financeMeanPrevDays',
 'appCat.financePrevDay',
 'appCat.gameGradient',
 'appCat.gameMeanPrevDays',
 'appCat.gamePrevDay',
 'appCat.officeGradient',
 'appCat.officeMeanPrevDays',
 'appCat.officePrevDay',
 'appCat.otherGradient',
 'appCat.otherMeanPrevDays',
 'appCat.otherPrevDay',
 'appCat.socialGradient',
 'appCat.socialMeanPrevDays',
 'appCat.socialPrevDay',
 'appCat.travelGradient',
 'appCat.travelMeanPrevDays',
 'appCat.travelPrevDay',
 'appCat.unknownGradient',
 'appCat.unknownMeanPrevDays',
 'appCat.unknownPrevDay',
 'appCat.utilitiesGradient',
 'appCat.utilitiesMeanPrevDays',
 'appCat.utilitiesPrevDay',
 'appCat.weatherGradient',
 'appCat.weatherMeanPrevDays',
 'appCat.weatherPrevDay',
 'circumplex.arousalGradient',
 'circumplex.arousalMeanPrevDays',
 'circumplex.arousalPrevDay',
 'circumplex.valenceGradient',
 'circumplex.valenceMeanPrevDays',
 'circumplex.valencePrevDay',
 'screenGradient',
 'screenMeanPrevDays',
 'screenPrevDay',
 'moodGradient',
 'moodMeanPrevDays',
 'moodPrevDay',
 'callGradient',
 'callSumPrevDays',
 'callPrevDay',
 'smsGradient',
 'smsSumPrevDays',
 'smsPrevDay']

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict

def retdata(a_range):
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
    ptcl3['weekday'] = pd.to_datetime(ptcl3['datepart']).dt.weekday_name
    le = LabelEncoder()
    ptcl3['weekday'] = le.fit_transform(ptcl3['weekday'])
    
    new_feature_list = ['id', 'datepart','weekday', 'mood_']
    for feature_name in ptcl3.columns:
        if feature_name not in ['id', 'datepart', 'sms', 'call','weekday']:
            new_feature_list.append((feature_name + 'PrevDay'))
            new_feature_list.append((feature_name + 'MeanPrevDays'))
            new_feature_list.append((feature_name + 'Gradient'))
        elif feature_name not in ['id', 'datepart','weekday']:
            new_feature_list.append((feature_name + 'PrevDay'))
            new_feature_list.append((feature_name + 'SumPrevDays'))
            new_feature_list.append((feature_name + 'Gradient'))
    
    the_df = pd.DataFrame(np.asarray([new_feature_list]))
    the_df.columns = the_df.loc[0,:]
    the_df = the_df.drop(0)
    
    #add previous day's mood

    id_set = list(OrderedDict.fromkeys(ptcl3['id']))
    for number in a_range:
        for person in id_set:
            persondf = ptcl3[ptcl3['id'] == person]
            for feature_name in persondf.columns:
                if feature_name == 'mood':
                    persondf['mood_'] = ptcl3['mood'] #all original feature names will be removed, hence the new name
                if feature_name not in ['id','datepart', 'call', 'sms','weekday']:
                    persondf[str(feature_name)+'PrevDay'] = persondf[feature_name].shift(1)
                    persondf[str(feature_name)+'PrevDay'] = persondf[str(feature_name)+'PrevDay'].fillna(0)
                    persondf[str(feature_name)+'MeanPrevDays'] = persondf[str(feature_name)+'PrevDay'].rolling(number).mean()
                    persondf[str(feature_name)+'Gradient'] = np.gradient(persondf[str(feature_name)+'PrevDay'].rolling(number).mean())
                    persondf = persondf.drop(feature_name, 1)
                elif feature_name not in ['id', 'datepart','weekday']: #looking at the sum instead of the mean of the previous days for sms and call
                    persondf[str(feature_name)+'PrevDay'] = persondf[feature_name].shift(1)
                    persondf[str(feature_name)+'PrevDay'] = persondf[str(feature_name)+'PrevDay'].fillna(0)
                    persondf[str(feature_name)+'SumPrevDays'] = persondf[str(feature_name)+'PrevDay'].rolling(number).sum()
                    persondf[str(feature_name)+'Gradient'] = np.gradient(persondf[str(feature_name)+'PrevDay'].rolling(number).mean())
                    persondf = persondf.drop(feature_name, 1)                

            persondf = persondf[persondf['activityGradient'].notnull()] #arbritrary feature to remove the first 6 days
            the_df = the_df.append(persondf)

        # replace null with 0 and reindex
        cleandata = the_df[cols].fillna(0)
        cleandata.index = range(len(cleandata.values))
        
        mses = list()
        mses_sd = list()
        rmses = list()
        
        #because every iteration of the rolling range gets a different train and test set, 
        #i use an average of mses over several training and test sets
        for i in range(25):  
            x = cleandata[cleandata.columns[3:]]
            y = cleandata['mood_']
            x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.3)
            
            lin_reg = LinearRegression()
            lin_reg.fit(x_train, y_train)
            lin_prediction = lin_reg.predict(x_test)
        
            errors = y_test.subtract(lin_prediction)
            squared_errors = errors*errors
            mse = squared_errors.mean()
            mse_sd = squared_errors.std()
            rmse = math.sqrt(squared_errors.mean())
            
            mses.append(mse)
            mses_sd.append(mse_sd*mse_sd)
            rmses.append(rmse)
            
        print('\n\t\t' + 'Rolling: ' + str(number) +'\n')
        print('\tMSE:', sum(mses)/len(mses), 'SD', sum(mses_sd)/len(mses_sd))
        print('\tRMSE:', sum(rmses)/len(mses))
        


retdata(range(2,8))

