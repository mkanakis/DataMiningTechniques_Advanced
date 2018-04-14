
# coding: utf-8

# In[84]:


import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import ttest_ind
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict

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
    for person in id_set:
        persondf = ptcl3[ptcl3['id'] == person]
        for feature_name in persondf.columns:
            if feature_name == 'mood':
                persondf['mood_'] = persondf['mood'] #all original feature names will be removed, hence the new name
            if feature_name not in ['id','datepart', 'call', 'sms','weekday']:
                persondf[str(feature_name)+'PrevDay'] = persondf[feature_name].shift(1)
                persondf[str(feature_name)+'PrevDay'] = persondf[str(feature_name)+'PrevDay'].fillna(0)
                persondf[str(feature_name)+'MeanPrevDays'] = persondf[str(feature_name)+'PrevDay'].rolling(7).mean()
                persondf[str(feature_name)+'Gradient'] = np.gradient(persondf[str(feature_name)+'PrevDay'].rolling(7).mean())
                persondf = persondf.drop(feature_name, 1)
            elif feature_name not in ['id', 'datepart','weekday']: #looking at the sum instead of the mean of the previous days for sms and call
                persondf[str(feature_name)+'PrevDay'] = persondf[feature_name].shift(1)
                persondf[str(feature_name)+'PrevDay'] = persondf[str(feature_name)+'PrevDay'].fillna(0)
                persondf[str(feature_name)+'SumPrevDays'] = persondf[str(feature_name)+'PrevDay'].rolling(7).sum()
                persondf[str(feature_name)+'Gradient'] = np.gradient(persondf[str(feature_name)+'PrevDay'].rolling(7).mean())
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




def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

id_set = list(OrderedDict.fromkeys(dsNotNormalized['id']))
result = {'base': 0, 'model': 0, 'equal': 0}
for person in id_set:
    # fix random seed for reproducibility
    numpy.random.seed(7)
    # load the dataset
    dataframe = dsNotNormalized[dsNotNormalized['id'] == person][['mood_']]
    dataframe.index = range(len(dataframe.values))
    baseline_set = pd.DataFrame([['mood', 'mood_f']])
    baseline_set.columns = baseline_set.values[0]
    baseline_set = baseline_set.drop(0)
    baseline_set['mood'] = dataframe
    baseline_set['mood_f'] = dataframe.shift(-1)
    baseline_set = baseline_set[baseline_set['mood_f'].notnull()]
    baseline_set.index = range(len(baseline_set.values))
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    train_size = int(len(dataset) * 0.5)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    # reshape into X=t and Y=t+1
    look_back = 3
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(5, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=0)
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    print('\n\n\t\t' + person + ':')
    errors_t = trainY[0] - trainPredict[:,0]
    squared_errors_t = errors_t*errors_t
    print('\nTrain Score:')
    print('\tMSE:', round(squared_errors_t.mean(), 4), 'SD', round(squared_errors_t.std(), 4))
    print('\tRMSE:', round(math.sqrt(squared_errors_t.mean()), 4), 'SD:', round(math.sqrt(squared_errors_t.std()), 4))
    
    errors = testY[0] - testPredict[:,0]
    squared_errors = errors*errors
    print('\nTest Score:')
    print('\tMSE:', round(squared_errors.mean(), 4), 'SD', round(squared_errors.std(), 4))
    print('\tRMSE:', round(math.sqrt(squared_errors.mean()), 4), 'SD:', round(math.sqrt(squared_errors.std()), 4))

    errors_b = baseline_set['mood'][train_size:len(dataset)] - baseline_set['mood_f'][train_size:len(dataset)]
    squared_errors_b = errors_b*errors_b
    print('\nBaseline Score:')
    print('\tMSE:', round(squared_errors_b.mean(), 4), 'SD', round(squared_errors_b.std(), 4))
    print('\tRMSE:', round(math.sqrt(squared_errors_b.mean()), 4), 'SD:', round(math.sqrt(squared_errors_b.std()), 4))
    
    t, p = ttest_ind(squared_errors_b, squared_errors, equal_var = False)
    print('\n\t\tT-test ||t:', t.round(4), '\tp:', p.round(4))
    if p < .2:
        if squared_errors_b.mean() < squared_errors.mean():
            result['base'] += 1
        else:
            result['model'] += 1
    else:
        result['equal'] += 1
            
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()
    
print(result)

