
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


x = dsNotNormalized[dsNotNormalized.columns[3:]]
y = dsNotNormalized['mood_']
x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.3)

#hyperparameter selection
tree_ = DecisionTreeRegressor()
forest_ = RandomForestRegressor()
xgb = XGBRegressor()

tuned_parameters_tree = [{'min_samples_leaf': range(35,46,5),'max_depth': range(45,66,5), 'max_features': range(50, 56, 1)}]
tuned_parameters_forest = [{'n_estimators': range(20,31,5), 'min_samples_leaf': range(10,31,5),'max_depth': range(10,21,5), 'max_features': range(45,52, 2)}]
tuned_parameters_xgb = [{'max_depth': range(1, 5, 2), 'n_estimators': range(70, 86, 5), 'learning_rate': [.125, .15, .175]}]


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
        print('\n\tT-test on mses of models (model vs. baseline)||t:', t, '\tp:, p)

