# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 11:35:50 2017

@author: Administrator
"""
import rf
import gbdt
import svc
import adbst
import numpy as np
import csv
import lg
import pandas as pd

if __name__=='__main__':
    test_ids,ret1,w1=rf.Titanic_rf()
    test_ids,ret2,w2=gbdt.Titanic_gbdt()
    test_ids,ret3,w3=svc.Titanic_svc()
    test_ids,ret4,w4=adbst.Titanic_adbst()
    test_ids,ret5,w5=lg.Titanic_lg()
    ret1=np.where(ret1==1,1,-1)
    ret2=np.where(ret2==1,1,-1)
    ret3=np.where(ret3==1,1,-1)
    ret4=np.where(ret4==1,1,-1)
    ret5=np.where(ret5==1,1,-1)
    votes=(w1+0.05)*ret1+w2*ret2+w3*ret3+w4*ret4+w5*ret5
    votes=np.where(votes<=0,0,1)
    '''
    submission=np.asarray(zip(test_ids,votes)).astype(int)
    #ensure passenger IDs in ascending order
    output=submission[submission[:,0].argsort()]
    predict_file=open("predict.csv",'wb')
    file_object=csv.writer(predict_file)
    file_object.writerow(["PassengerId","Survived"])
    file_object.writerows(output)
    predict_file.close()
    '''
    ids = pd.read_csv("predict.csv")
    submission_df = pd.DataFrame(data = {'PassengerId':ids["PassengerId"],'Survived':votes})
    print (submission_df.head(10))
    submission_df.to_csv('submission_combine.csv',columns = ['PassengerId','Survived'],index = False)
    print ('Done')
    print ('Done')
