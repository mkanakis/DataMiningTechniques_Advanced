# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 14:40:14 2018

@author: sabaa
"""


# coding: utf-8

# In[ ]:


from sklearn.linear_model import LinearRegression
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split as tts
from scipy.stats import ttest_ind
import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
import pandas as pd



x = dsNormalizedWhole[dsNormalizedWhole.columns[3:]]
y = dsNormalizedWhole['mood_']
pca = PCA(n_components=5)
ica = FastICA(n_components=5, max_iter=1000)
tsvd = TruncatedSVD(n_components=5)
gp = GaussianRandomProjection(n_components=5)
sp = SparseRandomProjection(n_components=5, dense_output=True)

x_pca = pd.DataFrame(pca.fit_transform(x))
x_ica = pd.DataFrame(ica.fit_transform(x))
x_tsvd = pd.DataFrame(tsvd.fit_transform(x))
x_gp = pd.DataFrame(gp.fit_transform(x))
x_sp = pd.DataFrame(sp.fit_transform(x))
x_pca.columns = ["pca_{}".format(i) for i in x_pca.columns]
x_ica.columns = ["ica_{}".format(i) for i in x_ica.columns]
x_tsvd.columns = ["tsvd_{}".format(i) for i in x_tsvd.columns]
x_gp.columns = ["gp_{}".format(i) for i in x_gp.columns]
x_sp.columns = ["sp_{}".format(i) for i in x_sp.columns]
X = pd.concat((x, x_pca), axis=1)
X = pd.concat((x, x_ica), axis=1)
X = pd.concat((x, x_tsvd), axis=1)
X = pd.concat((x, x_gp), axis=1)
X = pd.concat((x, x_sp), axis=1)
X['moodPrevDay_'] = dsNotNormalized[['moodPrevDay']]

x_train, x_test, y_train, y_test = tts(X, y, test_size = 0.2)

#x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.4)
x_base = x_test['moodPrevDay_']
x_test = x_test.drop('moodPrevDay_', 1)
x_train = x_train.drop('moodPrevDay_', 1)
print(x_train.shape)

#hyperparameter selection
tree_ = DecisionTreeRegressor()
forest_ = RandomForestRegressor()
xgb = XGBRegressor()

tuned_parameters_tree = [{'min_samples_leaf': range(35,46,5),'max_depth': range(45,66,5), 'max_features': range(50, 56, 1)}]
tuned_parameters_forest = [{'n_estimators': range(20,31,5), 'min_samples_leaf': range(10,31,5),'max_depth': range(10,21,5), 'max_features': range(45,52, 2)}]
tuned_parameters_xgb = [{'max_depth': range(2, 15, 2), 'n_estimators': [300], 'learning_rate': np.arange(0.01,0.16,0.02).tolist(),'colsample_bytree':np.arange(0.4,0.91,0.15).tolist(),'subsample':np.arange(0.3,0.8,0.1).tolist()}]


#models = [(tuned_parameters_tree, tree_, 'tree'), (tuned_parameters_forest, forest_, 'forest'), (tuned_parameters_xgb, xgb, 'xgb')]
models = [(tuned_parameters_forest, forest_, 'forest'), (tuned_parameters_xgb, xgb, 'xgb')]

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

lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
lin_prediction = lin_reg.predict(x_test)
predictionlist.append(('linear', lin_prediction))        
        
base_prediction = x_test['moodPrevDay']
predictionlist.append(('baseline', base_prediction)) 

for predictor, prediction in predictionlist:
    
    errors = y_test.subtract(prediction)
    squared_errors = errors*errors
    print('\n\t\t' + predictor.upper() + '\n')
    print('\tMSE:', squared_errors.mean(), 'SD', squared_errors.std())
    print('\tRMSE:', math.sqrt(squared_errors.mean()), 'SD:', math.sqrt(squared_errors.std()))
    
    errors_b = y_test.subtract(base_prediction)
    squared_errors_b = errors_b*errors_b
    
    if predictor not in ['baseline']:
        t, p = ttest_ind(squared_errors,squared_errors_b, equal_var=False)
        print('\n\tT-test on mses of models (model vs. baseline)||t:', t, '\tp:', p)
