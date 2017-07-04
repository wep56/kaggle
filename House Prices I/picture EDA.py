# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 15:31:40 2017

@author: Administrator
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import seaborn as sns
from sklearn.model_selection import KFold
from IPython.display import HTML, display
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.stats as st

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

quantitative = [f for f in train.columns if train.dtypes[f] != "object"]       #筛选出分类数据  
quantitative.remove('SalePrice')
quantitative.remove('Id')
qualitative = [f for f in train.columns if train.dtypes[f] == "object"]        #筛选出连续数据

###################缺失值查询#####################
missing = train.isnull().sum()
missing = missing[missing>0 ]
missing.sort_values(inplace=True)
missing.plot.bar()


total = train.isnull().sum().sort_values(ascending=False)
total = total[total>0]
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False) #特征值中缺失值占比
percent =percent[percent>0]
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.plot.bar()
missing_data.head(10)
"""
#################################################
#######################查询SalePrice分布######################
y = train["SalePrice"]
plt.figure(1); plt.title('Normal')                                             #st.norm 正态分布
sns.distplot(y,kde=False,fit=st.norm)
plt.figure(2); plt.title('Log Normal')                                         
sns.distplot(y,kde=False,fit=st.lognorm)                                       #st.lognorm 对数正态分布，对数据log处理，
                                                                               # 可以让数据变得平滑
print("Skewnesss：{}".format(y.skew()))                    #检测正态分布情况，使用峰度，偏度指数
print("Kurtosis：{}".format(y.kurt()))                     #峰度反映了峰部的尖度，

###################################################################
#####################连续数据分布情况分析###########################
f = pd.melt(train,value_vars =quantitative)                #利用pandas melt把特征单列出来
g = sns.FacetGrid(f,col="variable",col_wrap=2,sharex=False,sharey=False)  #各种不同的图，柱形图，添加密度函数
g = g.map(sns.distplot,"value")

###################分类数据箱形图#################################

for c in qualitative:
    train[c] = train[c].astype("category")                #把分类数据格式改成category
    if train[c].isnull().any():                           #分类数据中的缺失值用“Missing”填补
        train[c] = train[c].cat.add_categories(['Missing']) 
        train[c] = train[c].fillna("Missing")

def boxplot(x,y,**kwargs):
    sns.boxplot(x=x,y=y)
    x = plt.xticks(rotation=90)
f = pd.melt(train,id_vars=["SalePrice"],value_vars=qualitative)                #箱形图
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(boxplot, "value","SalePrice")

#########################分类数据：一元方差分析############################

def anova(frame):
    anv = pd.DataFrame()
    anv["feature"] = qualitative
    pvals = []
    for c in qualitative:
        samples = []
        for cls in frame[c].unique():
            s = frame[frame[c] == cls]["SalePrice"].values
            samples.append(s)
        pval = stats.f_oneway(*samples)[1]
        pvals.append(pval)
    anv["pval"] = pvals
    return anv.sort_values("pval")

a = anova(train)
a['disparity'] = np.log(1./a['pval'].values)
sns.barplot(data=a, x='feature', y='disparity')
x=plt.xticks(rotation=90)

############################相关性检验#####################
def encode(frame,feature):
    ordering = pd.DataFrame()
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val
    ordering['spmean'] = frame[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice']
    ordering = ordering.sort_values('spmean')
    ordering["ordering"] = range(1,ordering.shape[0]+1)
    ordering = ordering["ordering"].to_dict()
    
    for cat,o in ordering.items():
        frame.loc[frame[feature] == cat, feature+'_E'] = o

qual_encoded = []
for q in qualitative:  
    encode(train, q)
    qual_encoded.append(q+'_E')
print(qual_encoded)


def spearman(frame,features):
    spr = pd.DataFrame()
    spr["feature"] = features
    spr["spearman"] = [frame[f].corr(frame["SalePrice"],"spearman") for f in features]
    spr = spr.sort_values("spearman")
    plt.figure(figsize=(6,0.25*len(features)))
    sns.barplot(data=spr, y='feature', x='spearman', orient='h')
    
features = quantitative + qual_encoded
spearman(train, features)

#####################################################################################    

plt.figure(1)
corr = train[quantitative+["SalePrice"]].corr()
sns.heatmap(corr)
plt.figure(2)
corr = train[qual_encoded+['SalePrice']].corr()
sns.heatmap(corr)
plt.figure(3)
corr = pd.DataFrame(np.zeros([len(quantitative)+1,len(qual_encoded)+1]),index=quantitative+['SalePrice'], columns=qual_encoded+['SalePrice'])

for q1 in quantitative+["SalePrice"]:
    for q2 in qual_encoded+['SalePrice']:
        corr.loc[q1,q2] = train[q1].corr(train[q2])
sns.heatmap(corr)
    
###################################检查连续数据对预测数据的影响################   
features = quantitative

standard = train[train['SalePrice'] < 200000]
pricey = train[train['SalePrice'] >= 200000]

diff = pd.DataFrame()
diff['feature'] = features
diff['difference'] = [(pricey[f].fillna(0.).mean() - standard[f].fillna(0.).mean())/(standard[f].fillna(0.).mean())
                      for f in features]

sns.barplot(data=diff, x='feature', y='difference')
x=plt.xticks(rotation=90)

##################################聚类+主成成分分析##############################
features = quantitative + qual_encoded
model = TSNE(n_components=2,random_state=0,perplexity=50)
x = train[features].fillna(0).values
tsne = model.fit_transform(x)

std = StandardScaler()
s = std.fit_transform(x)
pca = PCA(n_components=30)
pca.fit(s)
pc = pca.transform(s)
kmeans = KMeans(n_clusters=5)
kmeans.fit(pc)
fr = pd.DataFrame({'tsne1': tsne[:,0], 'tsne2': tsne[:, 1], 'cluster': kmeans.labels_})
sns.lmplot(data=fr, x='tsne1', y='tsne2', hue='cluster', fit_reg=False)
print(np.sum(pca.explained_variance_ratio_))
"""
#####################################################################################

def scatter(v=None):
    var =v 
    data = pd.concat([train["SalePrice"],train[var]],axis=1)
    data.plot.scatter(x=var,y="SalePrice",ylim=(0,800000))

t = ['TotalBsmtSF','GrLivArea']

def boxplot(t=None):
    var = t
    data = pd.concat([train["SalePrice"],train[t]],axis=1)
    f,ax = plt.subplots(figsize=(8,6))
    fig = sns.boxplot(x=var,y="SalePrice",data=data)
    fig.axis(ymin=0,ymax=800000)
    
t = ['OverallQual','YearBuilt']

#######################    
    
corrmat = train.corr()
f,ax = plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=.8, square=True)

def heatmap(t=None):
    if t == None:
        corrmat = train.corr()
        f,ax =plt.subplots(figsize=(12,9))
        sns.heatmap(corrmat, vmax=.8, square=True)
    else:
        k = t
        corrmat = train.corr()
        cols = corrmat.nlargest(k,"SalePrice")["SalePrice"].index
        cm = np.corrcoef(train[cols].values.T)
        sns.set(font_scale=1.25)
        hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt=".2f",annot_kws={"size":k},yticklabels=cols.values, xticklabels=cols.values)
        plt.show()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

def pairplot(t):
    sns.pairplot(train[t], size = 2.5)
    plt.show()

    
    
    