# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 09:01:42 2017

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
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

df_train = pd.read_csv('train.csv', index_col='Id')
df_test = pd.read_csv('test.csv', index_col='Id')

#categorical features
print("Categorical:",df_train.select_dtypes(include=["object"]).columns)
#numerical features (see comment about 'MSSubCLass' here above)
print("Numerical:",df_train.select_dtypes(exclude=["object"]).columns)

"""
corrmat = df_train.corr()
plt.figure(figsize=(8,6))

k = 10
cols = corrmat.nlargest(k,"SalePrice")["SalePrice"].index
cm = np.corrcoef(df_train[cols].values.T)

#generate mask for upper triangle
mask = np.zeros_like(cm,dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.set(font_scale=1.25)
sns.heatmap(cm,mask=mask,cbar=True,annot=True,square=True,
            fmt=".2f",annot_kws={"size":10},yticklabels=cols.values,
            xticklabels=cols.values)
plt.show()     
#Let's plot 'GrLivArea', 'GarageArea', '1stFlrSF', 'TotalBsmtSF' and eyeball their distributions.
to_be_plotted = ['GrLivArea', 'GarageArea', '1stFlrSF', 'TotalBsmtSF']

plt.figure(figsize=(10,8))

for i,entry in enumerate(to_be_plotted):
    plt.subplot(2,2,i+1)
    plt.scatter(df_train[entry],df_train["SalePrice"])
    plt.xlabel(entry)
    plt.ylabel("SalePrice")

plt.tight_layout()
plt.show()

# TRAIN
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis=1,keys=['Total', 'Percent'])
missing_data.head(6)

# TEST
total = df_test.isnull().sum().sort_values(ascending=False)
percent = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis=1,keys=['Total', 'Percent'])
missing_data.head(6)

#Outliers
scaled_features = StandardScaler().fit_transform(df_train[['GrLivArea', 'GarageArea', \
                                                           '1stFlrSF', 'TotalBsmtSF']])
df_standard = pd.DataFrame(scaled_features, index=df_train.index, columns=['GrLivArea', \
                                                                           'GarageArea', '1stFlrSF', 'TotalBsmtSF'])
to_be_checked = ['GrLivArea', 'GarageArea', '1stFlrSF', 'TotalBsmtSF']

for entry in to_be_checked:
    print(entry.upper())
    s = df_standard[entry]
    print(s[s>s.std() *3])

dataset = pd.concat([df_train, df_test])

#1stFlrSF: First Floor square feet
#2ndFlrSF: Second floor square feet
#BsmtFinSF1: Type 1 finished square feet
#BsmtFinSF2: Type 2 finished square feet
#BsmtUnfSF: Unfinished square feet of basement area
#EnclosedPorch: Enclosed porch area in square feet
#GarageArea: Size of garage in square feet
#GrLivArea: Above grade (ground) living area square feet
#LotArea: Lot size in square feet
#LotFrontage: Linear feet of street connected to property
#MasVnrArea: Masonry veneer area in square feet
#MiscVal: $Value of miscellaneous feature
#OpenPorchSF: Open porch area in square feet
#PoolArea: Pool area in square feet
#ScreenPorch: Screen porch area in square feet
#TotalBsmtSF: Total square feet of basement area
#WoodDeckSF: Wood deck area in square feet

to_be_plotted = ['1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'EnclosedPorch',\
                'GarageArea', 'GrLivArea', 'LotArea', 'LotFrontage', 'MasVnrArea', 'MiscVal',\
                'OpenPorchSF', 'PoolArea', 'ScreenPorch', 'TotalBsmtSF', 'WoodDeckSF']


def plot_dsitribution(to_be_plotted):
    plt.figure(figsize=(25,125))
    for i,entry in enumerate(to_be_plotted):
        plt.subplot(18,2,i+1)
        dataset[entry].fillna(0,inplace=True)
        mu = dataset[entry].mean()
        sigma = dataset[entry].std()
        num_bins = 50
        n,bins,patches = plt.hist(dataset[entry],num_bins,normed=1)
        y = mlab.normpdf(bins,mu,sigma)
        plt.plot(bins, y, 'r--', linewidth=2)
        plt.xlabel(entry)
        plt.ylabel('Probability density')
        plt.grid(True)
    plt.show()
    
plot_dsitribution(to_be_plotted)


for entry in to_be_plotted:
    print(entry, ':', dataset[entry].skew())

for entry in to_be_plotted:
    if dataset[entry].skew() > 3:
        dataset[entry] = np.log1p(dataset[entry])
    
plot_dsitribution(to_be_plotted)
"""

def preprocessing(df_train,df_test):
    '''
    Parameters
    ----------
    df_train : pandas Dataframe
    df_test : pandas Dataframe
    
    Return
    ----------
    df_train : pandas Dataframe
    df_test : pandas Dataframe
    y_train : pandas Series
    '''
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
    
    #LotShape   General shape of property
    #LandContour Flatness of the property
    #LandSlope Slope of property
    #Electrical Electrical system
    #GarageType Garage location
    #PavedDrive Paved driveway
    #MiscFeature Miscellaneous feature not covered in other categories
    #YearRemodAdd Remodel date (same as construction date if no remodeling or additions)
    #YearBuilt Original construction date
    #YrSold Year Sold (YYYY)
    
    
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
    #面基为主
    continuous_vars = ['1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'EnclosedPorch',\
                'GarageArea', 'GrLivArea', 'LotArea', 'LotFrontage', 'MasVnrArea', 'MiscVal',\
                'OpenPorchSF', 'PoolArea', 'ScreenPorch', 'TotalBsmtSF', 'WoodDeckSF']
    skew_threshold = 3
    for entry in continuous_vars:
        if dataset[entry].skew() > skew_threshold:
            dataset[entry] = np.log1p(dataset[entry])
    
    
    ### standardization ###
    # standardization for continuous variables
    sub_df = dataset[continuous_vars]
    array_standard = StandardScaler().fit_transform(sub_df)
    df_standard = pd.DataFrame(array_standard, dataset.index, continuous_vars)
    dataset.drop(dataset[continuous_vars], axis=1, inplace=True)
    dataset = pd.concat([dataset, df_standard], axis=1)
    
    
    ### dummies ###
    # split back to training and test set
    df_train_len = len(df_train)
    df_dummies =  pd.get_dummies(dataset)
    df_train = df_dummies[:df_train_len]
    df_test = df_dummies[df_train_len:]

    return df_train, df_test, y_train
    

df_train, df_test, y_train_before_split = preprocessing(df_train, df_test)

# 80/20 split for df_train
validation_size = 0.2
seed = 3
X_train, X_validation, y_train, y_validation = train_test_split(df_train, y_train_before_split, \
            test_size=validation_size, random_state=seed)
X_test = df_test
"""
estimators = [("LassCV",LassoCV()),("RidgeCV",RidgeCV()),
              ('RandomForest', RandomForestRegressor()), ('GradientBoosting', GradientBoostingRegressor())]

for estimator in estimators:
    scores = cross_val_score(estimator=estimator[1],
                            X=X_train,
                            y=y_train,
                            scoring='r2',
                            cv=3,
                            n_jobs=-1)
    #print('CV accuracy scores: %s' % scores)
    print(estimator[0], 'CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    
#LassoCV
gs = GridSearchCV(
                estimator=LassoCV(),
                param_grid={'eps':[10**-7, 10**-5, 10**-3],
                            'n_alphas':[25, 50, 75]},
                scoring='r2',
                cv=3,
                n_jobs=-1)

gs = gs.fit(X_train, y_train)
print('LassoCV:')
print('Training accuracy: %.3f' % gs.best_score_)
print(gs.best_params_)
est = gs.best_estimator_
est.fit(X_train, y_train)
print('Best alpha: ', est.alpha_)
print('Validation accuracy: %.3f' % est.score(X_validation, y_validation))

#RidgeCV
gs = GridSearchCV(
                estimator=RidgeCV(),
                param_grid={'fit_intercept':[True, False],
                            'normalize':[True, False]},
                scoring='r2',
                cv=3,
                n_jobs=-1)

gs = gs.fit(X_train, y_train)
print('RidgeCV:')
print('Training accuracy: %.3f' % gs.best_score_)
print(gs.best_params_)
est = gs.best_estimator_
est.fit(X_train, y_train)
print('Best alpha: ', est.alpha_)
print('Validation accuracy: %.3f' % est.score(X_validation, y_validation))

    
#RandomForest
gs = GridSearchCV(
                estimator=RandomForestRegressor(random_state=seed),
                param_grid={'max_depth':[3, 10, 20],
                            'n_estimators':[10, 30, 50]},
                scoring='r2',
                cv=3,
                n_jobs=-1)

gs = gs.fit(X_train, y_train)
print('Random Forest:')
print('BR: ', gs.best_score_)
print('BR: ', gs.best_params_)
est = gs.best_estimator_
est.fit(X_train, y_train)
print('Validation accuracy: %.3f' % est.score(X_validation, y_validation))

#GradientBoosting
gs = GridSearchCV(
                estimator=GradientBoostingRegressor(random_state=seed),
                param_grid={'max_depth':[3, 10],
                            'learning_rate':[0.1, 0.03],
                            'n_estimators':[100, 250, 500]},
                scoring='r2',
                cv=3,
                n_jobs=-1)

gs = gs.fit(X_train, y_train)
print('Gradient Boosting:')
print('BR: ', gs.best_score_)
print('BR: ', gs.best_params_)
est = gs.best_estimator_
est.fit(X_train, y_train)
print('Validation accuracy: %.3f' % est.score(X_validation, y_validation))


model = LassoCV(eps=10**-7, n_alphas=75)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
submission = pd.DataFrame({"Id": df_test.index.values, "SalePrice": np.expm1(predictions)})
submission.to_csv('LassoCV_eps1e-07_nAlphas75.csv', index=False)
"""