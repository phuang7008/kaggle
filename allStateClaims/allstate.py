import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing

import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))

fair_constant = 0.7
def fair_obj(preds, dtrain):
    labels = dtrain.get_label()
    x = (preds - labels)
    den = abs(x) + fair_constant
    grad = fair_constant * x / (den)
    hess = fair_constant * fair_constant / (den * den)
    return grad, hess


shift = 200
RANDOM_STATE = 2012

# read in data in both training and test set
train = pd.read_csv("../input/train.csv")#, nrows=100)
test  = pd.read_csv("../input/test.csv")#,  nrows=100)

train_ids = train['id']
test_ids  = test['id']
y = np.log(train['loss'] + shift)
test['loss'] = np.nan
test_loss = test['loss']

trainX = train.drop(['id', 'loss'], 1)
testX  = test.drop(['id', 'loss'], 1)

combined = pd.concat([trainX, testX], axis=0)
print(combined.shape)

combined_cat = combined.select_dtypes(include=['object'])
combined_num = combined.select_dtypes(exclude=['object'])

combined_num['cont1_2_5'] = combined_num['cont1'] + combined_num['cont2'] + combined_num['cont5']
combined_num['cont4_7_10'] = combined_num['cont4'] + combined_num['cont7'] + combined_num['cont10']
combined_num['cont6_8_11'] = combined_num['cont6'] + combined_num['cont8'] + combined_num['cont11']
combined_num['cont3_13_14'] = combined_num['cont3'] + combined_num['cont13'] + combined_num['cont14']
combined_num['cont9_12'] = combined_num['cont9'] + combined_num['cont12'] 
#combined_num.drop(['cont1', 'cont2', 'cont3', 'cont4', 'cont5', 'cont6', 'cont7'], 1, inplace=True)
#combined_num.drop(['cont8', 'cont9', 'cont10', 'cont11', 'cont12', 'cont13', 'cont14'], 1, inplace=True)

for cat in combined_cat.columns:
    combined_cat[cat], unique = pd.factorize(combined_cat[cat])

joined = pd.concat([combined_cat, combined_num], axis=1)
print("joined shape is", joined.shape)

X = joined.iloc[:train_ids.shape[0],:]
X_test = joined.iloc[train_ids.shape[0]:,:]

# prepare data for XGBoost
dtrain = xgb.DMatrix(X, label=y)
dtest  = xgb.DMatrix(X_test)

params = { 'min_child_weight': 1,
           'eta': 0.01,
           'colsample_bytree': 0.7,
           'max_depth': 12,
           'subsample': 0.8,
           'alpha': 1,
           'gamma': 1,
           'silent': 1,
           'verbose_eval': True,
           'seed': RANDOM_STATE
         }

#params = {
#            'seed': 0,
#            'colsample_bytree': 0.7,
#            'silent': 1,
#            'subsample': 0.7,
#            'learning_rate': 0.03,
#            'objective': 'reg:linear',
#            'max_depth': 12,
#            'min_child_weight': 100,
#            'booster': 'gbtree'}

print("Before modeling")
del train
del test
del combined

clf = xgb.train(params, dtrain, 7500, feval=evalerror, maximize=False, obj=fair_obj)
prediction = np.exp(clf.predict(dtest)) - shift

submission = pd.DataFrame()
submission['id'] = test_ids
submission['loss'] = prediction
submission.to_csv('sub_v.csv', index=False)

import pickle
with open("xgboost_peter_7500.pkl", 'wb') as f:
    pickle.dump(clf, f)

print("Done!")
