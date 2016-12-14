import pandas as pd
import numpy as np
import xgboost as xgb

target_cols = ['ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1',
               'ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1',
               'ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1',
               'ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1',
               'ind_nom_pens_ult1','ind_recibo_ult1']

# load data
trainX = np.load("trainX.dat")
trainY = np.load("trainY.dat")
testX  = np.load("testX.npy")
print("train X is: ", trainX.shape)
print("train Y is: ", trainY.shape)
print("testX is: ", testX.shape)

# parameters
params = {'seed': 125,
          'colsample_bytree': 0.7,
          'silent': 1,
          'subsample': 0.7,
          'eta': 0.05,
          'objective': 'multi:softprob',
          'max_depth': 8,
          'min_child_weight': 1,
          'eval_metric': 'mlogloss',
          'num_class' : 22
         }
num_rounds = 1000

dtrain = xgb.DMatrix(trainX, label=trainY)
dtest  = xgb.DMatrix(testX)

model = xgb.train(params, dtrain, num_rounds)

del trainX, trainY

preds = model.predict(dtest)
del testX

target_cols = np.array(target_cols)
preds = np.argsort(preds, axis=1)
preds = np.fliplr(preds)[:,:7]
test_ids = np.array(np.load("test_ids.npy"))

final_preds = [" ".join(list(target_cols[pred])) for pred in preds]
out_df = pd.DataFrame({'ncodpers':test_ids, 'added_products':final_preds})
out_df.to_csv('submission.csv', index=False)

