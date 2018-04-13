"""
Created on Tue Apr 10 11:53:40 2018
@author: Saba, Mick
"""

#the preferred order for the DataFrame (just for inspection)
cols = ['id',
 'datepart',
 'weekday',
 'mood_',
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
from sklearn import preprocessing
from collections import OrderedDict
#import datetime
from matplotlib import pyplot
import xgboost
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
import sklearn

#normalizes the whole dataset
def normalize(df):
    
    result = df.copy()
    for feature_name in df.columns:
        if feature_name not in ['mood','mood_','circumplex.valence','circumplex.arousal','id','datepart','weekday']:
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
    ptcl3['weekday'] = pd.to_datetime(ptcl3['datepart']).dt.weekday_name
    le = sklearn.preprocessing.LabelEncoder()
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
    for person in id_set:
        persondf = ptcl3[ptcl3['id'] == person]
        for feature_name in persondf.columns:
            if feature_name == 'mood':
                persondf['mood_'] = ptcl3['mood'] #all original feature names will be removed, hence the new name
            if feature_name not in ['id','datepart', 'call', 'sms','weekday']:
                persondf[str(feature_name)+'PrevDay'] = persondf[feature_name].shift(1)
                persondf[str(feature_name)+'PrevDay'] = persondf[str(feature_name)+'PrevDay'].fillna(0)
                persondf[str(feature_name)+'MeanPrevDays'] = persondf[str(feature_name)+'PrevDay'].rolling(5).mean()
                persondf[str(feature_name)+'Gradient'] = np.gradient(persondf[str(feature_name)+'PrevDay'].rolling(5).mean())
                persondf = persondf.drop(feature_name, 1)
            elif feature_name not in ['id', 'datepart','weekday']: #looking at the sum instead of the mean of the previous days for sms and call
                persondf[str(feature_name)+'PrevDay'] = persondf[feature_name].shift(1)
                persondf[str(feature_name)+'PrevDay'] = persondf[str(feature_name)+'PrevDay'].fillna(0)
                persondf[str(feature_name)+'SumPrevDays'] = persondf[str(feature_name)+'PrevDay'].rolling(5).sum()
                persondf[str(feature_name)+'Gradient'] = np.gradient(persondf[str(feature_name)+'PrevDay'].rolling(5).mean())
                persondf = persondf.drop(feature_name, 1)                
                
        persondf = persondf[persondf['activityGradient'].notnull()] #arbritrary feature to remove the first 6 days
        the_df = the_df.append(persondf)
    
    # replace null with 0 and reindex
    cleandata = the_df[cols].fillna(0)
    cleandata.index = range(len(cleandata.values))
    
    #clean up
    del ptcl1, ptcl2, ptcl3, pt, pt1, pt2, df, df2, means
    #get normalized datasets
    normalizedwholeds = normalize(cleandata)
    normalizedperuser = normalizeperuser(cleandata)    
    
    return normalizedwholeds,normalizedperuser, cleandata


dsNormalizedWhole, dsNormalizedPerUser, dsNotNormalized = retdata()

Y = dsNormalizedWhole['mood_']
X = dsNormalizedWhole.drop(['mood_','id','datepart'],axis=1)
model = XGBRegressor()
model.fit(X.values, Y.values)
print(model.feature_importances_)
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()
#-------

plot_importance(model)
pyplot.show()
plot_importance(model,max_num_features=10)
pyplot.show()
#-------
X_train, X_test, y_train, y_test = train_test_split(X.values, Y.values, test_size=0.2, random_state=7)
#params={'eta':0.1,'seed':0,'subsample':0.8,'colsample_bytree':0.8,'objective':'reg:linear','max_depth':3,'min_child_weight':1}

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
#accuracy = accuracy_score(y_test, predictions)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))
## Fit model using each importance as a threshold
#thresholds = np.sort(model.feature_importances_)
#for thresh in thresholds:
#	# select features using threshold
#	selection = SelectFromModel(model, threshold=thresh, prefit=True)
#	select_X_train = selection.transform(X_train)
#	# train model
#	selection_model = XGBClassifier()
#	selection_model.fit(select_X_train, y_train)
#	# eval model
#	select_X_test = selection.transform(X_test)
#	y_pred = selection_model.predict(select_X_test)
#	predictions = [round(value) for value in y_pred]
#	accuracy = accuracy_score(y_test, predictions)
#	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
