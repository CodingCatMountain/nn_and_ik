#!/usr/bin/env python3
#coding=utf-8
from numpy.core.fromnumeric import reshape
import pandas as pd
import numpy as np
import os,sys
import sklearn.preprocessing
os.chdir(sys.path[0])

########################################
# @ Description: Check the data have the repeat one or not
#                Besides, the data should be scales to [-1,1] interval
#######################################

### empty the list x_ , y_ , z_
x_ = []; y_ = []; theta_ = []

if __name__ =="__main__":
    df = pd.read_csv('../data.csv')



    #flag = df.duplicated(subset=['x','y','theta'])
    flag = df.duplicated(subset=['theta'])
    print(flag)
    print(flag.sum())   # return is 0 . all data have no repeat .
    #print(df.to_string())
    

    for row in df.itertuples():
        x_.append(row.x);y_.append(row.y);theta_.append(row.theta)
    

    # transfer the list to ndarray
    x_ = np.array(x_);y_ = np.array(y_); theta_ = np.array(theta_)
    x_=reshape(x_,(1,7001));y_ = reshape(y_,(1,7001));theta_ = reshape(theta_,(1,7001))
    print(x_)


    # normalized
    x_normalized = sklearn.preprocessing.normalize(x_,norm="l2",axis=1)   ## When you normalize the data should pay attention to axis
    y_normalized = sklearn.preprocessing.normalize(y_,norm="l2",axis=1)
    theta_normalized = sklearn.preprocessing.normalize(theta_,norm="l2",axis=1)
    print(x_normalized)     

    dict = {"x_normalized":list(x_normalized[0]),\
            "y_normalized":list(y_normalized[0]),\
            "theta_normalized":list(theta_normalized[0])
    }

    df2 = pd.DataFrame(dict)
    dataframe = df.join(df2)
    dataframe.to_csv('./data_normalized.csv',index=False,mode='w',sep=',')
