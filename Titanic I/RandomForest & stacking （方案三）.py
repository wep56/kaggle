# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 13:38:26 2017

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 22:20:54 2016

@author: ReshamSarkar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import KFold

titanic_train = pd.read_csv("train.csv", dtype={"Age": np.float64}, )
titanic_test = pd.read_csv("test.csv", dtype={"Age": np.float64}, )

train_set = titanic_train.drop("Survived",axis=1)
df_combo = pd.concat((train_set,titanic_test),axis=0,ignore_index=True)

#  Embarked 缺失值用 C代替
df_combo["Embarked"] = df_combo["Embarked"].fillna("C")  

#存放 尊称
Title_list = pd.DataFrame(index=df_combo.index,columns=["Title"])
#存放 姓名
Surname_list = pd.DataFrame(index=df_combo.index,columns=["Surname"])
Name_list = list(df_combo.Name)
NL_1 = [elem.split("\n") for elem in Name_list]
ctr = 0
for j in NL_1:
    FullName = j[0]
    FullName = FullName.split(",")
    Surname_list.loc[ctr,"Surname"] = FullName[0]
    FullName = FullName.pop(1)
    FullName = FullName.split(".")
    FullName = FullName.pop(0)
    FullName = FullName.replace(" ","")
    Title_list.loc[ctr, "Title"] = str(FullName)
    ctr = ctr +1

#Title and Surname Extraction

Title_Dictionary = {
"Capt": "Officer",
"Col": "Officer",
"Major": "Officer",
"Jonkheer": "Sir",
"Don": "Sir",
"Sir" : "Sir",
"Dr": "Dr",
"Rev": "Rev",
"theCountess": "Lady",
"Dona": "Lady",
"Mme": "Mrs",
"Mlle": "Miss",
"Ms": "Mrs",
"Mr" : "Mr",
"Mrs" : "Mrs",
"Miss" : "Miss",
"Master" : "Master",
"Lady" : "Lady"
}   

def Title_Label(s):
    return Title_Dictionary[s]
df_combo["Title"] = Title_list["Title"].apply(Title_Label)

# 家族人数
Surname_Fam = pd.concat([Surname_list,df_combo[["SibSp","Parch"]]],axis=1)
Surname_Fam["Fam"] = Surname_Fam.Parch + Surname_Fam.SibSp + 1
Surname_Fam = Surname_Fam.drop(["SibSp", "Parch"], axis = 1)
df_combo = pd.concat([df_combo,Surname_Fam],axis=1)

#处理 Cabin
Cabin_List = df_combo.loc[:,["Cabin"]]
Cabin_List = Cabin_List.fillna("UNK")
#提取Cabin 中的首字母
Cabin_Code = []
for j in Cabin_List.Cabin:
    Cabin_Code.append(j[0])

Cabin_Code = pd.DataFrame({"Deck" : Cabin_Code})
df_combo = pd.concat([df_combo, Cabin_Code], axis = 1)

#对家族人数进行拆分 2<=x<=4 | 4<x<=7 & x=1 | s >7 三段
def Fam_label(s):
    if (s >= 2)&(s <=4):
        return 2
    elif ((s >4)&(s <=7)) | (s==1):
        return 1
    elif (s >7):
        return 0

df_combo["Fam"] = df_combo.loc[:,"Fam"].apply(Fam_label)

#对名字进行拆分 分成 “Royalty” | "Officer" |  s
def Title_label(s):
    if (s == "Sir") | ( s == "Lady"):
        return "Royalty"
    elif (s == "Dr") | (s == "Officer") | (s == "Rev"):
        return "Officer"
    else:
        return s
df_combo["Title"] = df_combo.loc[:,"Title"].apply(Title_label)  

#对船票进行处理 去除 , / 及空格。
def tix_clean(j):
    j = j.replace(".","")
    j = j.replace("/","")
    j = j.replace(" ","")
    return j
df_combo[["Ticket"]] = df_combo.loc[:,"Ticket"].apply(tix_clean)

# 计算处理后船票出现的频率
Ticket_count = dict(df_combo.Ticket.value_counts())

def Tix_ct(y):
    return Ticket_count[y]
df_combo["TicketGrp"] = df_combo.Ticket.apply(Tix_ct)

def Tix_label(s):
    if(s >=2)&(s <=4):
        return 2
    elif ((s >4) &(s <=8)) | (s ==1):
        return 1
    elif (s > 8):
        return 0

df_combo["TicketGrp"] = df_combo.loc[:,"TicketGrp"].apply(Tix_label)

df_combo.drop(["PassengerId", "Name", "Ticket", "Surname", "Cabin", "Parch", "SibSp"], axis=1, inplace = True)

mask_Age = df_combo.Age.notnull()
Age_Sex_Title_Pclass = df_combo.loc[mask_Age, ["Age", "Title", "Sex", "Pclass"]]
Filler_Ages = Age_Sex_Title_Pclass.groupby(by = ["Title", "Pclass", "Sex"]).median()
Filler_Ages = Filler_Ages.Age.unstack(level = -1).unstack(level = -1)

mask_Age = df_combo.Age.isnull()
Age_Sex_Title_Pclass_missing = df_combo.loc[mask_Age, ["Title", "Sex", "Pclass"]]

def Age_filler(row):
    if row.Sex == "female":
        age = Filler_Ages.female.loc[row["Title"],row["Pclass"]]
        return age
    elif row.Sex == "male":
        age = Filler_Ages.female.loc[row["Title"],row["Pclass"]]
        return age

Age_Sex_Title_Pclass_missing["Age"]  = Age_Sex_Title_Pclass_missing.apply(Age_filler, axis = 1)   

df_combo["Age"] = pd.concat([Age_Sex_Title_Pclass["Age"], Age_Sex_Title_Pclass_missing["Age"]])   

dumdum = (df_combo.Embarked == "S") & (df_combo.Pclass == 3)
df_combo.fillna(df_combo[dumdum].Fare.median(),inplace=True)

df_combo = pd.get_dummies(df_combo)


df_train = np.array(df_combo.loc[0:len(titanic_train["Survived"])-1])
df_test = np.array(df_combo.loc[len(titanic_train["Survived"]):])
"""
total_number_param = len(df_train.columns)
"""
df_target = titanic_train.Survived




select = SelectKBest(k =20)
clf = RandomForestClassifier(random_state=10,warm_start=True,
                             n_estimators =26,
                             max_depth=6,
                             max_features="sqrt")
pipeline = make_pipeline(select,clf)

pipeline.fit(df_train, df_target)
predictions = pipeline.predict(df_train)
predict_proba = pipeline.predict_proba(df_train)[:,1]

cv_score = cross_validation.cross_val_score(pipeline,df_train,df_target,cv=10)
print("Accuracy %.4g" %metrics.accuracy_score(df_target.values,predictions))
print("AUC Score (Train): %f"%metrics.roc_auc_score(df_target,predict_proba))
print("CV Score:Mean - %.7g|Std - %.7g |Min -%.7g |Max - %.7g "%(np.mean(cv_score),
                                                                 np.std(cv_score),
                                                                 np.min(cv_score),
                                                                 np.max(cv_score)))
final_pred = pipeline.predict(df_test)
submission = pd.DataFrame({"PassengerId": titanic_test["PassengerId"], "Survived": final_pred })
submission.to_csv("RandomForest_v2.csv", index=False) 

 
############################################################################################ 
#                                                 stacking                                          #
############################################################################################        
ntrain = df_train.shape[0]
ntest = df_test.shape[0]
SEED = 0
NFOLDS = 5
kf = KFold(ntrain,n_folds=NFOLDS,random_state=SEED)

class SklearnHelper(object):
    def __init__(self,clf,seed=0,params=None):
        params["random_state"] = seed
        self.clf = clf(**params)
    def train(self,x_train,y_train):
        self.clf.fit(x_train,y_train)
    
    def predict(self,x):
        return self.clf.predict(x)

def get_oof(clf, x_train, y_train, x_test):
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

# Assign the parameters for each of our 4 base models
rf_params = {
    'n_jobs': -1,
    'n_estimators': 575,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 3 
}
et_params = {
    'n_jobs': -1,
    'n_estimators':575,
    #'max_features': 0.5,
    'max_depth': 5,
    'min_samples_leaf': 3,
    'verbose': 3
}
ada_params = {
    'n_estimators': 575,
    'learning_rate' : 0.95
}

gb_params = {
    'n_estimators': 575,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 3,
    'verbose': 3
}
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }

rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
y_train = df_target.ravel()


# Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof(et, df_train, y_train, df_test)
rf_oof_train, rf_oof_test = get_oof(rf,df_train, y_train, df_test)
ada_oof_train, ada_oof_test = get_oof(ada, df_train, y_train, df_test)
gb_oof_train, gb_oof_test = get_oof(gb,df_train, y_train, df_test)
svc_oof_train, svc_oof_test = get_oof(svc,df_train, y_train, df_test)

x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)
print("{},{}".format(x_train.shape, x_test.shape))

#########################################################################################

rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test)
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test)
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test)
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test)

x_train1 = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test1 = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)

##############################################################################################
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

et_oof_train, et_oof_test = get_oof(et, x_train1, y_train, x_test1)
rf_oof_train, rf_oof_test = get_oof(rf,x_train1, y_train, x_test1)
ada_oof_train, ada_oof_test = get_oof(ada, x_train1, y_train, x_test1)
gb_oof_train, gb_oof_test = get_oof(gb,x_train1, y_train, x_test1)
svc_oof_train, svc_oof_test = get_oof(svc,x_train1, y_train, x_test1)

x_train2 = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test2 = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)


gb = GradientBoostingClassifier(
    #learning_rate = 0.02,
    n_estimators=575,
     #'max_features': 0.2,
    max_depth= 5,
    min_samples_leaf=3,
    verbose=3
 ).fit(x_train2, y_train)
y_final = gb.predict(x_test)
submission = pd.DataFrame({"Id": titanic_test.index.values, "SalePrice": y_final})
submission.to_csv('stacking4.csv', index=False)





