"""
Created on Tue Apr 10 11:53:40 2018
@author: Saba, Mick
"""

#the preferred order for the DataFrame (just for inspection)
cols = ['id',
 'datepart',
 'mood_',
 'weekday',
 'activityGradient',
 'activityMeanPrevDays',
 #'activityPrevDay',
 'appCat.builtinGradient',
 'appCat.builtinMeanPrevDays',
 #'appCat.builtinPrevDay',
 'appCat.communicationGradient',
 'appCat.communicationMeanPrevDays',
 #'appCat.communicationPrevDay',
 'appCat.entertainmentGradient',
 'appCat.entertainmentMeanPrevDays',
 #'appCat.entertainmentPrevDay',
 'appCat.financeGradient',
 'appCat.financeMeanPrevDays',
 #'appCat.financePrevDay',
 'appCat.gameGradient',
 'appCat.gameMeanPrevDays',
 #'appCat.gamePrevDay',
 'appCat.officeGradient',
 'appCat.officeMeanPrevDays',
 #'appCat.officePrevDay',
 'appCat.otherGradient',
 'appCat.otherMeanPrevDays',
 #'appCat.otherPrevDay',
 'appCat.socialGradient',
 'appCat.socialMeanPrevDays',
 #'appCat.socialPrevDay',
 'appCat.travelGradient',
 'appCat.travelMeanPrevDays',
 #'appCat.travelPrevDay',
 'appCat.unknownGradient',
 'appCat.unknownMeanPrevDays',
 #'appCat.unknownPrevDay',
 'appCat.utilitiesGradient',
 'appCat.utilitiesMeanPrevDays',
 #'appCat.utilitiesPrevDay',
 'appCat.weatherGradient',
 'appCat.weatherMeanPrevDays',
 #'appCat.weatherPrevDay',
 'circumplex.arousalGradient',
 'circumplex.arousalMeanPrevDays',
 #'circumplex.arousalPrevDay',
 'circumplex.valenceGradient',
 'circumplex.valenceMeanPrevDays',
 #'circumplex.valencePrevDay',
 'screenGradient',
 'screenMeanPrevDays',
 #'screenPrevDay',
 'moodGradient',
 'moodMeanPrevDays',
 #'moodPrevDay',
 'callGradient',
 'callSumPrevDays',
 #'callPrevDay',
 'smsGradient',
 'smsSumPrevDays']
 #'smsPrevDay']

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
from bayes_opt import BayesianOptimization
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection


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
    window_size=7
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
    ptcl3 = ptcl3.sort_values(by=['id','datepart'])
    
    new_feature_list = ['id', 'datepart','weekday', 'mood_']
    for feature_name in ptcl3.columns:
        if feature_name not in ['id', 'datepart', 'sms', 'call','weekday']:
            #new_feature_list.append((feature_name + 'PrevDay'))
            new_feature_list.append((feature_name + 'MeanPrevDays'))
            new_feature_list.append((feature_name + 'Gradient'))
            new_feature_list.append((feature_name + 'Log'))
        elif feature_name not in ['id', 'datepart','weekday']:
            #new_feature_list.append((feature_name + 'PrevDay'))
            new_feature_list.append((feature_name + 'SumPrevDays'))
            new_feature_list.append((feature_name + 'Gradient'))
    
    the_df = pd.DataFrame(np.asarray([new_feature_list]))
    the_df.columns = the_df.loc[0,:]
    the_df = the_df.drop(0)
    the_df = the_df.fillna(0)
    ptcl3 = ptcl3.fillna(0)
    
    #add previous day's mood

    id_set = list(OrderedDict.fromkeys(ptcl3['id']))
    for person in id_set:
        persondf = ptcl3[ptcl3['id'] == person]

        for feature_name in persondf.columns:
            if feature_name == 'mood':
                persondf['mood_'] = persondf['mood'] #all original feature names will be removed, hence the new name
            if feature_name not in ['id','datepart', 'call', 'sms','weekday']:
                #persondf[str(feature_name)+'PrevDay'] = persondf[feature_name].shift(1)
                #persondf[str(feature_name)+'PrevDay'] = persondf[str(feature_name)+'PrevDay'].fillna(0)
                persondf[str(feature_name)+'MeanPrevDays'] = persondf[str(feature_name)].rolling(window_size).mean()
                persondf[str(feature_name)+'Gradient'] = np.gradient(persondf[str(feature_name)].rolling(window_size).mean())
                persondf[str(feature_name)+'Log'] = np.log(persondf[str(feature_name)])
                persondf[str(feature_name)+'Log'][np.isneginf(persondf[str(feature_name)+'Log'])] = 0
                persondf = persondf.drop(feature_name, 1)
            elif feature_name not in ['id', 'datepart','weekday']: #looking at the sum instead of the mean of the previous days for sms and call
                #persondf[str(feature_name)+'PrevDay'] = persondf[feature_name].shift(1)
                #persondf[str(feature_name)+'PrevDay'] = persondf[str(feature_name)+'PrevDay'].fillna(0)
                persondf[str(feature_name)+'SumPrevDays'] = persondf[str(feature_name)].rolling(window_size).sum()
                persondf[str(feature_name)+'Gradient'] = np.gradient(persondf[str(feature_name)].rolling(window_size).mean())
                persondf = persondf.drop(feature_name, 1)                
                
        persondf = persondf[persondf['activityGradient'].notnull()] #arbritrary feature to remove the first 6 days
        persondf = persondf.fillna(0)
        
        pca = PCA(n_components=5)
        ica = FastICA(n_components=5, max_iter=1000)
        tsvd = TruncatedSVD(n_components=5)
        gp = GaussianRandomProjection(n_components=5)
        sp = SparseRandomProjection(n_components=5, dense_output=True)
        
        x_pca = pd.DataFrame(pca.fit_transform(persondf.drop(['mood_','id','datepart','weekday'],axis=1)))
        x_ica = pd.DataFrame(ica.fit_transform(persondf.drop(['mood_','id','datepart','weekday'],axis=1)))
        x_tsvd = pd.DataFrame(tsvd.fit_transform(persondf.drop(['mood_','id','datepart','weekday'],axis=1)))
        x_gp = pd.DataFrame(gp.fit_transform(persondf.drop(['mood_','id','datepart','weekday'],axis=1)))
        x_sp = pd.DataFrame(sp.fit_transform(persondf.drop(['mood_','id','datepart','weekday'],axis=1)))
        x_pca.columns = ["pca_{}".format(i) for i in x_pca.columns]
        x_ica.columns = ["ica_{}".format(i) for i in x_ica.columns]
        x_tsvd.columns = ["tsvd_{}".format(i) for i in x_tsvd.columns]
        x_gp.columns = ["gp_{}".format(i) for i in x_gp.columns]
        x_sp.columns = ["sp_{}".format(i) for i in x_sp.columns]
        X = pd.concat((persondf, x_pca), axis=1)
        X = pd.concat((persondf, x_ica), axis=1)
        X = pd.concat((persondf, x_tsvd), axis=1)
        X = pd.concat((persondf, x_gp), axis=1)
        X = pd.concat((persondf, x_sp), axis=1)

        the_df = the_df.append(persondf)
        the_df = the_df.fillna(0)
    
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

#redefine the model here and train it on the training set
model = XGBRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
from sklearn.metrics import mean_squared_error
import math
testScore=math.sqrt(mean_squared_error(y_test,y_pred))
print(testScore)
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
from sklearn.linear_model import LinearRegression
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split as tts

x = dsNormalizedWhole[dsNormalizedWhole.columns[3:]]
y = dsNormalizedWhole['mood_']
x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.2)

#hyperparameter selection
tree_ = DecisionTreeRegressor()
forest_ = RandomForestRegressor()
xgb = XGBRegressor()

tuned_parameters_tree = [{'min_samples_leaf': range(35,46,5),'max_depth': range(45,66,5), 'max_features': range(50, 56, 1)}]
tuned_parameters_forest = [{'n_estimators': range(20,31,5), 'min_samples_leaf': range(10,31,5),'max_depth': range(10,21,5), 'max_features': range(45,52, 2)}]
params={'eta':0.1,'seed':0,'subsample':0.8,'colsample_bytree':0.8,'objective':'reg:linear','max_depth':3,'min_child_weight':1}
tuned_parameters_xgb = [{'max_depth': range(3, 11, 1), 'n_estimators': 300, 'learning_rate': range(0.01,0.21,0.01),'colsample_bytree':range(0.4,0.91,0.1),'eta':range(0.05,0.31,0.01),'subsample':range(0.5,0.96,0.05),'eval_metric':'rmse'}]


models = [(tuned_parameters_tree, tree_, 'tree'), (tuned_parameters_forest, forest_, 'forest'), (tuned_parameters_xgb, xgb, 'xgb')]

best_models = list()

for tuned_parameters, model, modelname in models: 
    scores = ['neg_mean_squared_error']
    for score in scores:
        clf = GridSearchCV(model, tuned_parameters, cv=10, scoring=score)
        clf.fit(x_train, y_train)
       
        print('----------------------------------------------------------------------')
        print("Gridsearch for", modelname, "with", score, "as scoring method:")
        print()
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_, 'with score', clf.best_score_.round(3))
        print()
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print(mean.round(3), std.round(3), params)
        print()
        print()
        best_models.append((clf.best_score_.round(3), modelname, score, clf.best_params_))


predictionlist = list()
for modeldata in best_models:
    sme, model, error, params = modeldata
    if model == 'tree':
        treemodel = DecisionTreeRegressor(**params)
        treemodel.fit(x_train, y_train)
        tree_predict = treemodel.predict(x_test)
        predictionlist.append((model, tree_predict))
    if model == 'forest':
        forestmodel = RandomForestRegressor(**params)
        forestmodel.fit(x_train, y_train)
        forest_predict = forestmodel.predict(x_test)
        predictionlist.append((model, forest_predict))
    if model == 'xgb':
        xgbmodel = XGBRegressor(**params)
        xgbmodel.fit(x_train,y_train)
        xgb_predict = xgbmodel.predict(x_test)
        predictionlist.append((model, xgb_predict))

base_prediction = x_test['moodPrevDay']
predictionlist.append(('baseline', base_prediction)) 

lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
lin_prediction = lin_reg.predict(x_test)
predictionlist.append(('linear', lin_prediction))

for predictor, prediction in predictionlist:
    
    errors = y_test.subtract(prediction)
    squared_errors = errors*errors
    print('\n\t\t' + predictor.upper() + '\n')
    print('\tMSE:', squared_errors.mean(), 'SD', squared_errors.std())
    print('\tRMSE:', math.sqrt(squared_errors.mean()), 'SD:', math.sqrt(squared_errors.std()))
