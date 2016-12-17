import pandas as pd
import numpy as np

from collections import Counter
import warnings
warnings.filterwarnings('ignore')

target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1',
               'ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1',
               'ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1',
               'ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1',
               'ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']

train2015_05_28 = pd.read_csv("train2015_05_28.csv")
train2015_06_28 = pd.read_csv("train2015_06_28.csv")
train2015_06_28_unique = pd.read_csv("train2015_06_28_unique.csv")
train2015_05_28_unique = pd.read_csv("train2015_05_28_unique.csv")
train2016_05_28 = pd.read_csv("train2016_05_28.csv")
testX = pd.read_csv("testX.csv")

print("train2015_05_28 is ", train2015_05_28.shape)
print("train2015_06_28 is ", train2015_06_28.shape)
print("Customers are only in 05-28 (2015): ", train2015_05_28_unique.shape)
print("Customers are only in 06-28 (2015): ", train2015_06_28_unique.shape)
print("train2016_05_28 is ", train2016_05_28.shape)
print("testX is ", testX.shape)

# get unique customer IDs
train55customers = set(train2015_05_28['ncodpers'])
train56customers = set(train2015_06_28['ncodpers'])
print("2015-05-28 total customers: ", len(train55customers))
print("2015-06-28 total customers: ", len(train56customers))

# find customers in both 2015-05-28 and 2015-06-28
train_common_customers = train55customers & train56customers
print("Customers are in both 05-28 and 06-28 sets (2015): ", len(train_common_customers))

# need to find out those customers who changed services betweeen 2015-05-28 to 2015-06-28
may_data  = train2015_05_28.sort_values(by='ncodpers').reset_index(drop=True).set_index('ncodpers')[target_cols]
june_data = train2015_06_28.sort_values(by='ncodpers').reset_index(drop=True).set_index('ncodpers')[target_cols]
print(may_data.shape)
print(june_data.shape)
print(june_data[june_data.index==15889])

# need to find out those customers who changed services betweeen 2015-05-28 to 2015-06-28
# to use this approach, I need to make sure their index are the same
changed_status_55 = []

train_common_customers = list(train_common_customers)

for id in range(len(train_common_customers)):
    customer = train_common_customers[id]
    #print(customer)
    
    all_2015_05 = may_data[may_data.index == customer]
    all_2015_06 = june_data[june_data.index == customer]
    #print(all_2015_05.shape[0])

    # for services changed
    for idx1, row1 in all_2015_06.iterrows():
        for idx2, row2 in all_2015_05.iterrows():
            tmp = [max(int(x1) - int(x2),0) for (x1, x2) in zip(row1, row2)]
            if sum(tmp) > 0:
                changed_status_55.append(customer)
                
#    try:
#        assert_frame_equal(all_2015_05, all_2015_06)
#    except:
        #changed_status_55 = pd.concat([changed_status_55, all_2015_05])
        #changed_status_56 = pd.concat([changed_status_56, all_2015_06])
#        changed_status_55.append(customer)
#        changed_status_56.append(customer)
        
print("2015-05 total remaining: ", len(changed_status_55))
np.save("changed_ids_new2", changed_status_55)

train2015_05_28_changed = train2015_05_28[train2015_05_28.ncodpers.isin(changed_status_55)]
train2015_06_28_changed = train2015_06_28[train2015_06_28.ncodpers.isin(changed_status_55)]
print(train2015_05_28_changed.shape)
print(train2015_06_28_changed.shape)

# save them
train2015_05_28_changed.to_csv("train2015_05_28_changed.csv", index=False)
train2015_06_28_changed.to_csv("train2015_06_28_changed.csv", index=False)
