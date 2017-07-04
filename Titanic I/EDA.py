# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 15:09:40 2017

@author: Administrator
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("train.csv")
data["Age"].fillna(data["Age"].median(),inplace=True)
def missing():
    missing = data.isnull().sum()
    missing = missing[missing>0]
    missing.sort_values(inplace=True)
    missing.plot.bar()
    plt.xlabel('missing')
    plt.ylabel('Number of passengers')
    plt.legend()


def sex():  
    survived_sex = data[data["Survived"]==1]["Sex"].value_counts()
    died_sex =data[data["Survived"]==0]["Sex"].value_counts()
    df = pd.DataFrame([survived_sex,died_sex])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(4,4))
    plt.xlabel('Sex')
    plt.ylabel('Number of passengers')
    plt.legend()

def Pclass():
    survived_Pclass = data[data["Survived"]==1]["Pclass"].value_counts()
    died_Pclass =data[data["Survived"]==0]["Pclass"].value_counts()
    df = pd.DataFrame([survived_Pclass,died_Pclass])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(4,4))
    plt.xlabel('Pclass')
    plt.ylabel('Number of passengers')
    plt.legend()
    
def age():
    figure = plt.figure(figsize=(15,8))
    plt.hist([data[data['Survived']==1]['Age'], data[data['Survived']==0]['Age']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
    plt.xlabel('Age')
    plt.ylabel('Number of passengers')
    plt.legend()

def fare():
    figure = plt.figure(figsize=(15,8))
    plt.hist([data[data['Survived']==1]['Fare'], data[data['Survived']==0]['Fare']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
    plt.xlabel('Fare')
    plt.ylabel('Number of passengers')
    plt.legend()

def AgeFare():
    plt.figure(figsize=(15,8))
    ax = plt.subplot()
    ax.scatter(data[data['Survived']==1]['Age'],data[data['Survived']==1]['Fare'],c='green',s=40)
    ax.scatter(data[data['Survived']==0]['Age'],data[data['Survived']==0]['Fare'],c='red',s=40)
    ax.set_xlabel('Age')
    ax.set_ylabel('Fare')
    ax.legend(('survived','dead'),scatterpoints=1,loc='upper right',fontsize=15,)
    
def embark():
    survived_embark = data[data['Survived']==1]['Embarked'].value_counts()
    dead_embark = data[data['Survived']==0]['Embarked'].value_counts()
    df = pd.DataFrame([survived_embark,dead_embark])
    df.index = ['Survived','Dead']
    df.plot(kind='bar', stacked=True, figsize=(15,8))
    
def PclassFare():
    ax = plt.subplot()
    ax.set_ylabel('Average fare')
    data.groupby('Pclass').mean()['Fare'].plot(kind='bar',figsize=(15,8), ax = ax)
    
def sibsp():
    survived_sibsp = data[data['Survived']==1]['SibSp'].value_counts()
    dead_sibsp = data[data['Survived']==0]['SibSp'].value_counts()
    df = pd.DataFrame([survived_sibsp,dead_sibsp])
    df.index = ['Survived','Dead']
    df.plot(kind='bar', stacked=True, figsize=(15,8))
    
def parch():
    survived_parch = data[data['Survived']==1]['Parch'].value_counts()
    dead_parch = data[data['Survived']==0]['Parch'].value_counts()
    df = pd.DataFrame([survived_parch,dead_parch])
    df.index = ['Survived','Dead']
    df.plot(kind='bar', stacked=True, figsize=(15,8))

def family():
    data['FamilySize'] = data['Parch'] + data['SibSp'] + 1
    survived_FamilySize = data[data['Survived']==1]['FamilySize'].value_counts()
    dead_FamilySize = data[data['Survived']==0]['FamilySize'].value_counts()
    df = pd.DataFrame([survived_FamilySize,dead_FamilySize])
    df.index = ['Survived','Dead']
    df.plot(kind='bar', stacked=True, figsize=(15,8))

def child():
    data["child"] = np.where(data["Age"]<12,1,0)
    survived_child = data[data['Survived']==1]["child"].value_counts()
    dead_child = data[data['Survived']==0]["child"].value_counts()
    df = pd.DataFrame([survived_child,dead_child])
    df.index = ['Survived','Dead']
    df.plot(kind='bar', stacked=True, figsize=(15,8))
    
  
def allimage():
    missing()
    sex()
    Pclass()
    age()
    fare()
    AgeFare()
    embark()
    PclassFare()
    sibsp()
    parch()
    family()
    child()




    




