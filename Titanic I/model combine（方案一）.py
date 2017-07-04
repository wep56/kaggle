# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 17:50:22 2017

@author: Administrator
"""
import pandas as pd
import dataProcess
import numpy as np
import time
import csv
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.learning_curve import learning_curve
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def data():
    input_df, submit_df = dataProcess.getDataSets(bins=True, scaled=True,binary=True)
    submit_ids = submit_df['PassengerId']  
    input_df.drop('PassengerId', axis=1, inplace=1) 
    submit_df.drop('PassengerId', axis=1, inplace=1)
    
    features_list = input_df.columns.values[1::]
    x = input_df.ix[:, 1::]
    y = input_df.ix[:, 0]
    test = submit_df.values
    
    clf = ExtraTreesClassifier(n_estimators=200)
    clf = clf.fit(x, y)
    features = pd.DataFrame()
    features['feature'] = x.columns
    features['importance'] = clf.feature_importances_
    features.sort(['importance'],ascending=False)
    model = SelectFromModel(clf, prefit=True)
    
    train_new = model.transform(x)
    train_new.shape
    test_new = model.transform(test)
    test_new.shape
    return x,y,test,train_new,test_new
    

"""
input_df, submit_df = dataProcess.getDataSets(bins=True, scaled=True,binary=True)
submit_ids = submit_df['PassengerId']        
input_df.drop('PassengerId', axis=1, inplace=1) 
submit_df.drop('PassengerId', axis=1, inplace=1)
features_list = input_df.columns.values[1::]
x = input_df.ix[:, 1::]
y = input_df.ix[:, 0]
test = submit_df.values

clf = ExtraTreesClassifier(n_estimators=200)
clf = clf.fit(x, y)

features = pd.DataFrame()
features['feature'] = x.columns
features['importance'] = clf.feature_importances_
features.sort(['importance'],ascending=False)

model = SelectFromModel(clf, prefit=True)
train_new = model.transform(x)
train_new.shape

test_new = model.transform(test)
test_new.shape
"""
##########################0.78469########################################################

def rf():
    x,y,test,train_new,test_new = data()
    
    """
    input_df, submit_df = dataProcess.getDataSets(bins=True, scaled=True,binary=True)
    submit_ids = submit_df['PassengerId']        
    input_df.drop('PassengerId', axis=1, inplace=1) 
    submit_df.drop('PassengerId', axis=1, inplace=1)
    features_list = input_df.columns.values[1::]
    x = input_df.ix[:, 1::]
    y = input_df.ix[:, 0]
    test = submit_df.values
    clf = ExtraTreesClassifier(n_estimators=200)
    clf = clf.fit(x, y)
    features = pd.DataFrame()
    features['feature'] = x.columns
    features['importance'] = clf.feature_importances_
    features.sort(['importance'],ascending=False)
    model = SelectFromModel(clf, prefit=True)
    
    train_new = model.transform(x)
    train_new.shape
    test_new = model.transform(test)
    test_new.shape
    
    forest =RandomForestClassifier(max_features="sqrt")
    parameters ={"max_depth":[4,5,6,7,8],
             "n_estimators":[200,210,240,250],
             "criterion":["gini","entropy"]}
    cross_validation = StratifiedKFold(y, n_folds=5)
    grid_search=GridSearchCV(forest,param_grid=parameters,cv=cross_validation)
    grid_search.fit(train_new,y)
    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
    """
    parameters = {'criterion': 'gini', 'max_depth': 5, 'n_estimators': 200,"oob_score":True}
    forest =RandomForestClassifier(max_features="sqrt",**parameters)

    test_accs=[]
    for i in range(5):
        X_train,X_hold,y_train,y_hold=train_test_split(train_new,y,test_size=0.3)
        forest.fit(X_train,y_train)
        acc=forest.score(X_hold,y_hold)
        print ("\nAccuracy is:{:.4f}".format(acc))
        test_accs.append(acc)
    acc_mean="%.3f"%(np.mean(test_accs))
    acc_std="%.3f"%(np.std(test_accs))
    print ("\nmean accuracy:",acc_mean,"and stddev:",acc_std)
    ########################Step8:Predicting and Saving result######################################
    forest.fit(train_new,y)
    return forest.predict(test_new),float(acc_mean)
   

def lg():
    x,y,test,train_new,test_new = data()
    """
    input_df, submit_df = dataProcess.getDataSets(bins=True, scaled=True,binary=True)
    submit_ids = submit_df['PassengerId']        
    input_df.drop('PassengerId', axis=1, inplace=1) 
    submit_df.drop('PassengerId', axis=1, inplace=1)
    features_list = input_df.columns.values[1::]
    x = input_df.ix[:, 1::]
    y = input_df.ix[:, 0]
    test = submit_df.values
    clf = ExtraTreesClassifier(n_estimators=200)
    clf = clf.fit(x, y)
    features = pd.DataFrame()
    features['feature'] = x.columns
    features['importance'] = clf.feature_importances_
    features.sort(['importance'],ascending=False)
    model = SelectFromModel(clf, prefit=True)
    
    train_new = model.transform(x)
    train_new.shape
    test_new = model.transform(test)
    test_new.shape
    
    lg =LogisticRegression()
    parameters={"penalty":['l1','l2'],
             "C":[0.1,0.2,0.4,0.6,0.8,1,2,4,6,10],
              "tol":[0.0001,0.001,0.01,0.1],
              "random_state":[42,1234567890]}
    cross_validation = StratifiedKFold(y, n_folds=5)
    grid_search=GridSearchCV(lg,param_grid=parameters,cv=cross_validation)
    grid_search.fit(train_new,y)
    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
    """
    parameters = {'C': 6, 'penalty': 'l2', 'random_state': 42, 'tol': 0.01}
    lg =LogisticRegression(**parameters)

    test_accs=[]
    for i in range(5):
        X_train,X_hold,y_train,y_hold=train_test_split(train_new,y,test_size=0.3)
        lg.fit(X_train,y_train)
        acc=lg.score(X_hold,y_hold)
        print ("\nAccuracy is:{:.4f}".format(acc))
        test_accs.append(acc)
    acc_mean="%.3f"%(np.mean(test_accs))
    acc_std="%.3f"%(np.std(test_accs))
    print ("\nmean accuracy:",acc_mean,"and stddev:",acc_std)
    ########################Step8:Predicting and Saving result######################################
    lg.fit(train_new,y)
    return lg.predict(test_new),float(acc_mean)

def svc():
    x,y,test,train_new,test_new = data()
    """
    input_df, submit_df = dataProcess.getDataSets(bins=True, scaled=True,binary=True)
    submit_ids = submit_df['PassengerId']        
    input_df.drop('PassengerId', axis=1, inplace=1) 
    submit_df.drop('PassengerId', axis=1, inplace=1)
    features_list = input_df.columns.values[1::]
    x = input_df.ix[:, 1::]
    y = input_df.ix[:, 0]
    test = submit_df.values
    
    rbf_params = {"kernel": ['rbf'],
                    "class_weight": ["balanced"],
                    "C": [0.1,0.3,0.5,1,3,5],
                    "gamma": [0.01,0.05,0.1,0.5],
                    "tol": 10.0**-np.arange(2,4),
                    "random_state": [42,1234567890]}
    poly_params = {"kernel": ['poly'],
                    "class_weight": ["balanced"],
                    "degree": [1,3,5,7],                    #poly核专用 其他忽略
                    "C": [0.1,0.3,0.5,1,3,5],
                    "gamma": 10.0**np.arange(-1, 1),
                    "coef0": 10.0**-np.arange(1,2),          #poly/sigmoid核专用 其他忽略      
                    "tol": 10.0**-np.arange(1,3),
                    "random_state": [42,1234567890]}
   
    sigmoid_params = {"kernel": ['sigmoid'],
                        "class_weight": ["balanced"],
                        "C": 10.0**np.arange(-2,6),
                        "gamma": 10.0**np.arange(-3, 3),
                        "coef0": 10.0**-np.arange(1,5),       #poly/sigmoid核专用 其他忽略
                        "tol": 10.0**-np.arange(2,4),
                        "random_state": [42,1234567890]}
    #通过网格搜索选择“rbf”与“poly”核,经过偏差与方差还有精确度分析最终选择“poly”核.
    params_rbf = {"kernel": 'rbf',
                    "class_weight": 'balanced',
                    "C": 3,
                    "gamma": 0.1,
                    "tol": 0.01,
                    "random_state": 42}
    """
    params_poly = {"kernel": 'poly',
                "class_weight": 'balanced',
                 "degree":3,
                 "C": 3,
                 "gamma": 0.1,
                 "coef0": 0.1,
                 "tol": 0.01,
                 "random_state": 42}

    svc=SVC(**params_poly)
    test_accs=[]
    for i in range(5):
        X_train,X_hold,y_train,y_hold=train_test_split(x,y,test_size=0.3)
        svc.fit(X_train,y_train)
        acc=svc.score(X_hold,y_hold)
        print ("\nAccuracy is:{:.4f}".format(acc))
        test_accs.append(acc)
    acc_mean="%.3f"%(np.mean(test_accs))
    acc_std="%.3f"%(np.std(test_accs))
    print ("\nmean accuracy:",acc_mean,"and stddev:",acc_std)
    svc.fit(x,y)
    return svc.predict(test),float(acc_mean)
    

#######################################0.77#################################

def gbdt():
    x,y,test,train_new,test_new = data()
    """
    input_df, submit_df = dataProcess.getDataSets(bins=True, scaled=True,binary=True)
    submit_ids = submit_df['PassengerId']        
    input_df.drop('PassengerId', axis=1, inplace=1) 
    submit_df.drop('PassengerId', axis=1, inplace=1)
    features_list = input_df.columns.values[1::]
    x = input_df.ix[:, 1::]
    y = input_df.ix[:, 0]
    test = submit_df.values
    
    clf = ExtraTreesClassifier(n_estimators=200)
    clf = clf.fit(x, y)
    features = pd.DataFrame()
    features['feature'] = x.columns
    features['importance'] = clf.feature_importances_
    features.sort(['importance'],ascending=False)
    model = SelectFromModel(clf, prefit=True)
    
    train_new = model.transform(x)
    train_new.shape
    test_new = model.transform(test)
    test_new.shape
    
    clf=GradientBoostingClassifier(n_estimators=60,max_depth=15,max_features="sqrt",
                               min_samples_split=100,min_samples_leaf=70,subsample=0.85)
    parameters={"learning_rate":[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5]}
    print ("Hyperparameter opimization using RandomizedSearchCV...")
    cross_validation = StratifiedKFold(y, n_folds=5)
    grid_search=GridSearchCV(clf,param_grid=parameters,cv=cross_validation,scoring='roc_auc')
    grid_search.fit(train_new,y)
    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
    output = grid_search.predict(test_new).astype(int)
    """
    parameters ={"n_estimators":60,"max_depth":15,"max_features":"sqrt","min_samples_split":100,
             "min_samples_leaf":70,"subsample":0.85,"learning_rate":0.01}
    clf=GradientBoostingClassifier(**parameters)
    
    test_accs=[]
    for i in range(5):
        X_train,X_hold,y_train,y_hold=train_test_split(train_new,y,test_size=0.3)
        clf.fit(X_train,y_train)
        acc=clf.score(X_hold,y_hold)
        print ("\nAccuracy is:{:.4f}".format(acc))
        test_accs.append(acc)
    acc_mean="%.3f"%(np.mean(test_accs))
    acc_std="%.3f"%(np.std(test_accs))
    print ("\nmean accuracy:",acc_mean,"and stddev:",acc_std)
    clf.fit(train_new,y)
    return clf.predict(test_new),float(acc_mean)
########################################################################

######################################0.76#################################

def adbst():
    x,y,test,train_new,test_new = data()
    """
    input_df, submit_df = dataProcess.getDataSets(bins=True, scaled=True,binary=True)
    submit_ids = submit_df['PassengerId']        
    input_df.drop('PassengerId', axis=1, inplace=1) 
    submit_df.drop('PassengerId', axis=1, inplace=1)
    features_list = input_df.columns.values[1::]
    x = input_df.ix[:, 1::]
    y = input_df.ix[:, 0]
    test = submit_df.values
    clf = ExtraTreesClassifier(n_estimators=200)
    clf = clf.fit(x, y)
    features = pd.DataFrame()
    features['feature'] = x.columns
    features['importance'] = clf.feature_importances_
    features.sort(['importance'],ascending=False)
    model = SelectFromModel(clf, prefit=True)
    
    train_new = model.transform(x)
    train_new.shape
    test_new = model.transform(test)
    test_new.shape
    """
    parameters = {'C': 6, 'penalty': 'l2', 'random_state': 42, 'tol': 0.01}
    lg =LogisticRegression(**parameters)
    """
    clf=AdaBoostClassifier(lg,learning_rate=0.1,n_estimators=40,random_state=42)
    parameters={"algorithm":["SAMME","SAMME.R"]}
    print ("Hyperparameter opimization using RandomizedSearchCV...")
    cross_validation = StratifiedKFold(y, n_folds=5)
    grid_search=GridSearchCV(clf,param_grid=parameters,cv=cross_validation,scoring='roc_auc')
    grid_search.fit(train_new,y)
    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
    output = grid_search.predict(test_new).astype(int)
    """
    clf=AdaBoostClassifier(lg,learning_rate=0.001,n_estimators=1000,random_state=42)
    test_accs=[]
    for i in range(5):
        X_train,X_hold,y_train,y_hold=train_test_split(train_new,y,test_size=0.3)
        clf.fit(X_train,y_train)
        acc=clf.score(X_hold,y_hold)
        print ("\nAccuracy is:{:.4f}".format(acc))
        test_accs.append(acc)
    acc_mean="%.3f"%(np.mean(test_accs))
    acc_std="%.3f"%(np.std(test_accs))
    print ("\nmean accuracy:",acc_mean,"and stddev:",acc_std)
    clf.fit(train_new,y)
    return clf.predict(test_new),float(acc_mean)

######################################0.76#################################
"""
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)

params = [1,10,15,20,25,30,40]
test_scores = []
for param in params:
    bagging_clf = BaggingClassifier(clf, n_estimators=param, n_jobs=-1)
    test_score = np.sqrt(-cross_val_score(bagging_clf, train_new, y,scoring = 'neg_mean_squared_error' ,cv=5))
    test_scores.append(np.mean(test_score))

plt.plot(params,test_scores)
plt.title('n_estimators vs CV Error')
plt.show()    

br = BaggingClassifier(base_estimator = clf,n_estimators = 25)
br.fit(train_new,y)
y_final = np.expm1(br.predict(test_new))




def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, 
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")

        plt.legend(loc="best")

        plt.draw()
        plt.show()
        plt.gca().invert_yaxis()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff

plot_learning_curve(clf, u"学习曲线", X, y)
"""
if __name__ == "__main__":
    ret1,w1 = rf()
    ret2,w2 = lg()
    ret3,w3 = svc()
    ret4,w4 = gbdt()
    ret5,w5 = adbst()
    ret1 = np.where(ret1==1,1,-1)
    ret2 = np.where(ret2==1,1,-1)
    ret3 = np.where(ret3==1,1,-1)
    ret4 = np.where(ret4==1,1,-1)
    ret5 = np.where(ret5==1,1,-1)
    votes = (w1+0.01)*ret1 + w2*ret2 +w3*ret3 +w4*ret4+w5*ret5
    votes = np.where(votes <=0,0,1)
    
    ids = pd.read_csv("predict.csv")
    submission_df = pd.DataFrame(data = {'PassengerId':ids["PassengerId"],'Survived':votes})
    print (submission_df.head(10))
    submission_df.to_csv('combine6.csv',columns = ['PassengerId','Survived'],index = False)
    print ('Done')
    print ('Done')

    