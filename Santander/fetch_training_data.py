import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# read in all files
train2015_05_28 = pd.read_csv("train2015_05_28_changed.csv")
train2015_06_28 = pd.read_csv("train2015_06_28_changed.csv")
train2016_05_28 = pd.read_csv("train2016_05_28.csv")
testX = pd.read_csv("testX.csv")

print(train2015_05_28.shape)
print(train2015_06_28.shape)
print(train2016_05_28.shape)
print(testX.shape)

# the unique ones in either 2015-05-28 or 2015-06-28
train2015_05_28_unique = pd.read_csv('train2015_05_28_unique.csv')
print("Unique ones are: ", train2015_05_28_unique.shape)

train2015_06_28_unique = pd.read_csv('train2015_06_28_unique.csv')
print("Unique ones are: ", train2015_06_28_unique.shape)

droplist = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'fecha_dato', 'fecha_alta', 'ult_fec_cli_1t', 'conyuemp', 'tipodom']
train2015_05_28.drop(droplist, 1, inplace=True)
train2015_06_28.drop(droplist, 1, inplace=True)
train2016_05_28.drop(droplist, 1, inplace=True)
print(train2015_05_28.shape)
print(train2015_06_28.shape)
print(train2016_05_28.shape)

train2015_05_28_unique.drop(droplist, 1, inplace=True)
train2015_06_28_unique.drop(droplist, 1, inplace=True)
print("Unique ones are: ", train2015_05_28_unique.shape)
print("Unique ones are: ", train2015_06_28_unique.shape)

train2015_05_28.loc[train2015_05_28.antiguedad < 0, 'antiguedad'] = 0
train2015_06_28.loc[train2015_06_28.antiguedad < 0, 'antiguedad'] = 0
train2016_05_28.loc[train2016_05_28.antiguedad < 0, 'antiguedad'] = 0
print(train2015_05_28.shape)
print(train2015_06_28.shape)
print(train2016_05_28.shape)

train2015_05_28_unique.loc[train2015_05_28_unique.antiguedad < 0, 'antiguedad'] = 0
train2015_06_28_unique.loc[train2015_06_28_unique.antiguedad < 0, 'antiguedad'] = 0
print("Unique ones are: ", train2015_05_28_unique.shape)
print("Unique ones are: ", train2015_06_28_unique.shape)

# for padding
# need to add those adding in 2015-06-28
target_cols = ['ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1',
               'ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1',
               'ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1',
               'ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1',
               'ind_nom_pens_ult1','ind_recibo_ult1']

padding_for_55 = train2015_06_28_unique.copy()
padding_for_55[target_cols] = [0] * 22
train55 = pd.concat([train2015_05_28, padding_for_55])
train56 = pd.concat([train2015_06_28, train2015_06_28_unique])

# now need to find out changed target in a single column list format

trainY = []
trainX = []

for idx, row in train56.iterrows():
    customer_id = row.ncodpers
    row55 = train55[train55.ncodpers == customer_id]        
    cur_target_list  = row[target_cols].values
    prev_target_list = row55[target_cols].values[0]
    row55.drop(['ncodpers'], 1, inplace=True)
    row55 = row55.values[0]
    #print(prev_target_list, " and ", cur_target_list)

    new_target_list =[max(int(x1) - int(x2), 0) for (x1, x2) in zip(cur_target_list, prev_target_list)]
    
    for ind, val in enumerate(new_target_list):
        if val != 0:
            trainX.append(row55)
            trainY.append(ind)
            
print(len(trainY))
print(len(trainX))

from collections import Counter
print(Counter(trainY))

trainX = np.array(trainX)
trainY = np.array(trainY)
print(trainY.shape)
print(trainX.shape)
print(trainX[0])

trainY.dump("trainY.dat")
trainX.dump("trainX.dat")
