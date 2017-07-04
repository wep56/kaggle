# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 11:29:51 2017

@author: Administrator
"""
#import os
#mingw_path = r'C:/Program Files/mingw-w64/x86_64-7.1.0-posix-seh-rt_v5-rev0/mingw64/bin'
#os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV,Ridge
from sklearn.linear_model import LassoCV,Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,AdaBoostRegressor,ExtraTreesRegressor
import seaborn as sns
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn.svm import SVR
from math import sqrt

df_train = pd.read_csv('train.csv', index_col='Id')
df_test = pd.read_csv('test.csv', index_col='Id')

#categorical features
print('Categorical: ', df_train.select_dtypes(include=['object']).columns)

#numerical features (see comment about 'MSSubCLass' here above)
print('Numerical: ', df_train.select_dtypes(exclude=['object']).columns)

def preprocessing(df_train,df_test):
    # remove outliers in GrLivArea
    df_train.drop(df_train[df_train['GrLivArea'] > 4500].index, inplace=True)

    # Normalize SalePrice using log_transform
    y_train = np.log1p(df_train['SalePrice'])
    # Remove SalePrice from training and merge training and test data
    df_train.pop('SalePrice')
    dataset = pd.concat([df_train, df_test])

    # Numerical variable with "categorical meaning"
    # Cast it to str so that we get dummies later on
    dataset['MSSubClass'] = dataset['MSSubClass'].astype(str)

    
    ### filling NaNs ###
    # no alley
    dataset["Alley"].fillna("None", inplace=True)

    # no basement
    dataset["BsmtCond"].fillna("None", inplace=True)
    dataset["BsmtExposure"].fillna("None", inplace=True)
    dataset["BsmtFinSF1"].fillna(0, inplace=True)               
    dataset["BsmtFinSF2"].fillna(0, inplace=True)               
    dataset["BsmtUnfSF"].fillna(0, inplace=True)                
    dataset["TotalBsmtSF"].fillna(0, inplace=True)
    dataset["BsmtFinType1"].fillna("None", inplace=True)
    dataset["BsmtFinType2"].fillna("None", inplace=True)
    dataset["BsmtFullBath"].fillna(0, inplace=True)
    dataset["BsmtHalfBath"].fillna(0, inplace=True)
    dataset["BsmtQual"].fillna("None", inplace=True)

    # most common electrical system
    dataset["Electrical"].fillna("SBrkr", inplace=True)

    # one missing in test; set to other
    dataset["Exterior1st"].fillna("Other", inplace=True)
    dataset["Exterior2nd"].fillna("Other", inplace=True)

    # no fence
    dataset["Fence"].fillna("None", inplace=True)

    # no fireplace
    dataset["FireplaceQu"].fillna("None", inplace=True)

    # fill with typical functionality
    dataset["Functional"].fillna("Typ", inplace=True)

    # no garage
    dataset["GarageArea"].fillna(0, inplace=True)
    dataset["GarageCars"].fillna(0, inplace=True)
    dataset["GarageCond"].fillna("None", inplace=True)
    dataset["GarageFinish"].fillna("None", inplace=True)
    dataset["GarageQual"].fillna("None", inplace=True)
    dataset["GarageType"].fillna("None", inplace=True)
    dataset["GarageYrBlt"].fillna("None", inplace=True)

    # "typical" kitchen
    dataset["KitchenQual"].fillna("TA", inplace=True)

    # lot frontage (no explanation for NA values, perhaps no frontage)
    dataset["LotFrontage"].fillna(0, inplace=True)

    # Masonry veneer (no explanation for NA values, perhaps no masonry veneer)
    dataset["MasVnrArea"].fillna(0, inplace=True)
    dataset["MasVnrType"].fillna("None", inplace=True)

    # most common value
    dataset["MSZoning"].fillna("RL", inplace=True)

    # no misc features
    dataset["MiscFeature"].fillna("None", inplace=True)

    # description says NA = no pool, but there are entries with PoolArea >0 and PoolQC = NA. Fill the ones with values with average condition
    dataset.loc[(dataset['PoolQC'].isnull()) & (dataset['PoolArea']==0), 'PoolQC' ] = 'None'
    dataset.loc[(dataset['PoolQC'].isnull()) & (dataset['PoolArea']>0), 'PoolQC' ] = 'TA'

    # classify missing SaleType as other
    dataset["SaleType"].fillna("Other", inplace=True)

    # most common
    dataset["Utilities"].fillna("AllPub", inplace=True)

    
    ### feature engineering ###
    # create new binary variables: assign 1 to mode
    
    dataset["IsRegularLotShape"] = (dataset["LotShape"] == "Reg") * 1
    dataset["IsLandLevel"] = (dataset["LandContour"] == "Lvl") * 1
    dataset["IsLandSlopeGentle"] = (dataset["LandSlope"] == "Gtl") * 1
    dataset["IsElectricalSBrkr"] = (dataset["Electrical"] == "SBrkr") * 1
    dataset["IsGarageDetached"] = (dataset["GarageType"] == "Detchd") * 1
    dataset["IsPavedDrive"] = (dataset["PavedDrive"] == "Y") * 1
    dataset["HasShed"] = (dataset["MiscFeature"] == "Shed") * 1
    
    # was the house remodeled? if yes, assign 1
    dataset["Remodeled"] = (dataset["YearRemodAdd"] != dataset["YearBuilt"]) * 1
    # assign 1 to houses which were sold the same year they were remodeled
    dataset["RecentRemodel"] = (dataset["YearRemodAdd"] == dataset["YrSold"]) * 1
    # assign 1 to houses which were sold the same year they were built
    dataset["VeryNewHouse"] = (dataset["YearBuilt"] == dataset["YrSold"]) * 1
    

    
    ### normalization ###
    # normalize distribution for continuous variables with skew > 3
    continuous_vars = ['1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'EnclosedPorch',\
                'GarageArea', 'GrLivArea', 'LotArea', 'LotFrontage', 'MasVnrArea', 'MiscVal',\
                'OpenPorchSF', 'PoolArea', 'ScreenPorch', 'TotalBsmtSF', 'WoodDeckSF']
    
    skew_threshold = 3
    for entry in continuous_vars:
        if dataset[entry].skew() > skew_threshold:
            dataset[entry] = np.log1p(dataset[entry])
    
    sub_df = dataset[continuous_vars]
    array_standard = StandardScaler().fit_transform(sub_df)
    df_standard = pd.DataFrame(array_standard,dataset.index,continuous_vars)
    dataset.drop(dataset[continuous_vars],axis=1,inplace=True)
    dataset = pd.concat([dataset,df_standard],axis=1)
    
    ### dummies ###
    # split back to training and test set
    df_tranin_len = len(df_train)
    df_dummies = pd.get_dummies(dataset)
    df_train = df_dummies[:df_tranin_len]
    df_test = df_dummies[df_tranin_len:]
    
    return df_train, df_test, y_train





######################### 使用Bagging 得分0.11772 #############################
#大概alpha=10~20的时候，可以把score达到0.1100左右。

"""
df_train, df_test, y_train = preprocessing(df_train, df_test)
X_test = df_test

alphas = np.logspace(-3,2,50)
test_scores = []
for alpha in alphas:
    clf = Ridge(alpha)
    test_score = np.sqrt(-cross_val_score(clf,df_train,y_train,cv=10,scoring="neg_mean_squared_error"))
    test_scores.append(np.mean(test_score))

plt.plot(alphas,test_scores)
plt.title('Alpha vs CV Error')
plt.show()


max_features = [.1,.3,.5,.7,.9,.99]
test_scores = []
for max_feat in max_features:
    clf = RandomForestRegressor(n_estimators=200,max_features=max_feat)
    test_score = np.sqrt(-cross_val_score(clf,df_train,y_train,cv=5,scoring="neg_mean_squared_error"))
    test_scores.append(np.mean(test_score))

plt.plot(max_features,test_scores)
plt.title('Max Features vs CV Error')
plt.show()  
 
    
ridge = Ridge(alpha=15)

params = [1,10,15,20,25,30,40]
test_scores = []
for param in params:
    clf = BaggingRegressor(base_estimator = ridge,n_estimators = param)
    test_score = np.sqrt(-cross_val_score(clf,df_train,y_train,cv=5,scoring="neg_mean_squared_error"))
    test_scores.append(np.mean(test_score))

plt.plot(params,test_scores)
plt.title('n_estimators vs CV Error')
plt.show()

br = BaggingRegressor(base_estimator = ridge,n_estimators = 10)
br.fit(df_train,y_train)
y_final = np.expm1(br.predict(X_test))


submission = pd.DataFrame({"Id": df_test.index.values, "SalePrice": y_final})
submission.to_csv('Bagging1.csv', index=False)


params = [10,15,20,25,30,35,40,45,50]
test_scores = []
for param in params:
    clf = AdaBoostRegressor(base_estimator = ridge,n_estimators = param)
    test_score = np.sqrt(-cross_val_score(clf,df_train,y_train,cv=5,scoring="neg_mean_squared_error"))
    test_scores.append(np.mean(test_score))

plt.plot(params,test_scores)
plt.title('max_depth vs CV Error')
plt.show()

#################################### stacking 0.1362 ############################################

df_train = pd.read_csv('train.csv', index_col='Id')
df_test = pd.read_csv('test.csv', index_col='Id')

#categorical features
print('Categorical: ', df_train.select_dtypes(include=['object']).columns)

#numerical features (see comment about 'MSSubCLass' here above)
print('Numerical: ', df_train.select_dtypes(exclude=['object']).columns)

def preprocessing(df_train,df_test):
    # Normalize SalePrice using log_transform
    y_train = np.log1p(df_train['SalePrice'])
    # Remove SalePrice from training and merge training and test data
    df_train.pop('SalePrice')
    dataset = pd.concat([df_train, df_test])

    # Numerical variable with "categorical meaning"
    # Cast it to str so that we get dummies later on
    dataset['MSSubClass'] = dataset['MSSubClass'].astype(str)

    
    ### filling NaNs ###
    # no alley
    dataset["Alley"].fillna("None", inplace=True)

    # no basement
    dataset["BsmtCond"].fillna("None", inplace=True)
    dataset["BsmtExposure"].fillna("None", inplace=True)
    dataset["BsmtFinSF1"].fillna(0, inplace=True)               
    dataset["BsmtFinSF2"].fillna(0, inplace=True)               
    dataset["BsmtUnfSF"].fillna(0, inplace=True)                
    dataset["TotalBsmtSF"].fillna(0, inplace=True)
    dataset["BsmtFinType1"].fillna("None", inplace=True)
    dataset["BsmtFinType2"].fillna("None", inplace=True)
    dataset["BsmtFullBath"].fillna(0, inplace=True)
    dataset["BsmtHalfBath"].fillna(0, inplace=True)
    dataset["BsmtQual"].fillna("None", inplace=True)

    # most common electrical system
    dataset["Electrical"].fillna("SBrkr", inplace=True)

    # one missing in test; set to other
    dataset["Exterior1st"].fillna("Other", inplace=True)
    dataset["Exterior2nd"].fillna("Other", inplace=True)

    # no fence
    dataset["Fence"].fillna("None", inplace=True)

    # no fireplace
    dataset["FireplaceQu"].fillna("None", inplace=True)

    # fill with typical functionality
    dataset["Functional"].fillna("Typ", inplace=True)

    # no garage
    dataset["GarageArea"].fillna(0, inplace=True)
    dataset["GarageCars"].fillna(0, inplace=True)
    dataset["GarageCond"].fillna("None", inplace=True)
    dataset["GarageFinish"].fillna("None", inplace=True)
    dataset["GarageQual"].fillna("None", inplace=True)
    dataset["GarageType"].fillna("None", inplace=True)
    dataset["GarageYrBlt"].fillna("None", inplace=True)

    # "typical" kitchen
    dataset["KitchenQual"].fillna("TA", inplace=True)

    # lot frontage (no explanation for NA values, perhaps no frontage)
    dataset["LotFrontage"].fillna(0, inplace=True)

    # Masonry veneer (no explanation for NA values, perhaps no masonry veneer)
    dataset["MasVnrArea"].fillna(0, inplace=True)
    dataset["MasVnrType"].fillna("None", inplace=True)

    # most common value
    dataset["MSZoning"].fillna("RL", inplace=True)

    # no misc features
    dataset["MiscFeature"].fillna("None", inplace=True)

    # description says NA = no pool, but there are entries with PoolArea >0 and PoolQC = NA. Fill the ones with values with average condition
    dataset.loc[(dataset['PoolQC'].isnull()) & (dataset['PoolArea']==0), 'PoolQC' ] = 'None'
    dataset.loc[(dataset['PoolQC'].isnull()) & (dataset['PoolArea']>0), 'PoolQC' ] = 'TA'

    # classify missing SaleType as other
    dataset["SaleType"].fillna("Other", inplace=True)

    # most common
    dataset["Utilities"].fillna("AllPub", inplace=True)

    
    ### feature engineering ###
    # create new binary variables: assign 1 to mode
    dataset["IsRegularLotShape"] = (dataset["LotShape"] == "Reg") * 1
    dataset["IsLandLevel"] = (dataset["LandContour"] == "Lvl") * 1
    dataset["IsLandSlopeGentle"] = (dataset["LandSlope"] == "Gtl") * 1
    dataset["IsElectricalSBrkr"] = (dataset["Electrical"] == "SBrkr") * 1
    dataset["IsGarageDetached"] = (dataset["GarageType"] == "Detchd") * 1
    dataset["IsPavedDrive"] = (dataset["PavedDrive"] == "Y") * 1
    dataset["HasShed"] = (dataset["MiscFeature"] == "Shed") * 1
    # was the house remodeled? if yes, assign 1
    dataset["Remodeled"] = (dataset["YearRemodAdd"] != dataset["YearBuilt"]) * 1
    # assign 1 to houses which were sold the same year they were remodeled
    dataset["RecentRemodel"] = (dataset["YearRemodAdd"] == dataset["YrSold"]) * 1
    # assign 1 to houses which were sold the same year they were built
    dataset["VeryNewHouse"] = (dataset["YearBuilt"] == dataset["YrSold"]) * 1

    
    ### normalization ###
    # normalize distribution for continuous variables with skew > 3
    continuous_vars = ['1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'EnclosedPorch',\
                'GarageArea', 'GrLivArea', 'LotArea', 'LotFrontage', 'MasVnrArea', 'MiscVal',\
                'OpenPorchSF', 'PoolArea', 'ScreenPorch', 'TotalBsmtSF', 'WoodDeckSF']
    skew_threshold = 3
    for entry in continuous_vars:
        if dataset[entry].skew() > skew_threshold:
            dataset[entry] = np.log1p(dataset[entry])
    sub_df = dataset[continuous_vars]
    array_standard = StandardScaler().fit_transform(sub_df)
    df_standard = pd.DataFrame(array_standard,dataset.index,continuous_vars)
    dataset.drop(dataset[continuous_vars],axis=1,inplace=True)
    dataset = pd.concat([dataset,df_standard],axis=1)
    
    ### dummies ###
    # split back to training and test set
    df_dummies = pd.get_dummies(dataset)
    x_train = np.array(df_dummies[:df_train.shape[0]])
    x_test = np.array(df_dummies[df_train.shape[0]:])
    return x_train,x_test,y_train

x_train, x_test,y_train  = preprocessing(df_train,df_test)
"""
from scipy.stats import skew

TARGET = 'SalePrice'
NFOLDS = 5
SEED = 0
NROWS = None
SUBMISSION_FILE = '../input/sample_submission.csv'


## Load the data ##
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

ntrain = train.shape[0]
ntest = test.shape[0]

## Preprocessing ##

y_train = np.log(train[TARGET]+1)


train.drop([TARGET], axis=1, inplace=True)


all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))


#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data = pd.get_dummies(all_data)

#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())

x_train = np.array(all_data[:train.shape[0]])
x_test = np.array(all_data[train.shape[0]:])

ntrain = x_train.shape[0]
ntest = x_test.shape[0]

TARGET = "SalePrice"
NFOLDS =5
SEED = 0

kf = KFold(ntrain,n_folds = NFOLDS,shuffle=True,random_state=SEED)

class SklearnWrapper(object):
    def __init__(self,clf,seed=0,params=None):
        params["random_state"] = seed
        self.clf = clf(**params)
        
    def train(self,x_train,y_train):
        self.clf.fit(x_train,y_train)
        
    def predict(self,x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)
    
def get_oof(clf,x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


et_params = {
    'n_jobs': -1,
    'n_estimators': 100,
    'max_features': 0.5,
    'max_depth': 12,
    'min_samples_leaf': 2,
}

rf_params = {
    'n_jobs': -1,
    'n_estimators': 100,
    'max_features': 0.2,
    'max_depth': 12,
    'min_samples_leaf': 2,
}


rd_params={
    'alpha': 10,
}


ls_params={
    'alpha': 0.005
}

gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 3,
    'min_samples_leaf': 2,
    'verbose': 0
}

et = SklearnWrapper(clf=ExtraTreesRegressor, seed=SEED, params=et_params)
rf = SklearnWrapper(clf=RandomForestRegressor, seed=SEED, params=rf_params)
rd = SklearnWrapper(clf=Ridge, seed=SEED, params=rd_params)
ls = SklearnWrapper(clf=Lasso, seed=SEED, params=ls_params)
gb = SklearnWrapper(clf=GradientBoostingRegressor, seed=SEED, params=gb_params)

y_train = y_train.ravel()

et_oof_train,et_oof_test = get_oof(et, x_train, y_train, x_test)
rf_oof_train,rf_oof_test = get_oof(rf,x_train, y_train, x_test)
rd_oof_train,rd_oof_test = get_oof(rd,x_train, y_train, x_test)
ls_oof_train,ls_oof_test = get_oof(ls,x_train, y_train, x_test)
gb_oof_train,gb_oof_test = get_oof(gb,x_train, y_train, x_test)

print("Training is complete")

print("ET-CV: {}".format(sqrt(mean_squared_error(y_train, et_oof_train))))
print("RF-CV: {}".format(sqrt(mean_squared_error(y_train, rf_oof_train))))
print("RD-CV: {}".format(sqrt(mean_squared_error(y_train, rd_oof_train))))
print("LS-CV: {}".format(sqrt(mean_squared_error(y_train, ls_oof_train))))
print("gb-CV: {}".format(sqrt(mean_squared_error(y_train, gb_oof_train))))

x_train1 = np.concatenate((et_oof_train, rf_oof_train, rd_oof_train, ls_oof_train,gb_oof_train), axis=1)
x_test1 = np.concatenate((et_oof_test, rf_oof_test, rd_oof_test, ls_oof_test,gb_oof_test), axis=1)

et = SklearnWrapper(clf=ExtraTreesRegressor, seed=SEED, params=et_params)
rf = SklearnWrapper(clf=RandomForestRegressor, seed=SEED, params=rf_params)
rd = SklearnWrapper(clf=Ridge, seed=SEED, params=rd_params)
ls = SklearnWrapper(clf=Lasso, seed=SEED, params=ls_params)
gb = SklearnWrapper(clf=GradientBoostingRegressor, seed=SEED, params=gb_params)

y_train = y_train.ravel()

et_oof_train,et_oof_test = get_oof(et, x_train1, y_train, x_test1)
rf_oof_train,rf_oof_test = get_oof(rf,x_train1, y_train, x_test1)
rd_oof_train,rd_oof_test = get_oof(rd,x_train1, y_train, x_test1)
ls_oof_train,ls_oof_test = get_oof(ls,x_train1, y_train, x_test1)
gb_oof_train,gb_oof_test = get_oof(gb,x_train1, y_train, x_test1)

print("Training is complete")

print("ET-CV: {}".format(sqrt(mean_squared_error(y_train, et_oof_train))))
print("RF-CV: {}".format(sqrt(mean_squared_error(y_train, rf_oof_train))))
print("RD-CV: {}".format(sqrt(mean_squared_error(y_train, rd_oof_train))))
print("LS-CV: {}".format(sqrt(mean_squared_error(y_train, ls_oof_train))))
print("gb-CV: {}".format(sqrt(mean_squared_error(y_train, gb_oof_train))))

gb = GradientBoostingRegressor(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,                        
 subsample=0.8,
 ).fit(x_train, y_train)
y_final = np.expm1(gb.predict(x_test))
submission = pd.DataFrame({"Id": df_test.index.values, "SalePrice": y_final})
submission.to_csv('stacking3.csv', index=False)