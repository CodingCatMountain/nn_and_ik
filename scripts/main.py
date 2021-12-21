#!/usr/bin/env python3
#coding=utf-8
from math import tanh
import re
import numpy as np
import scipy 
import matplotlib.pyplot as plt
import pandas as pd
import os,sys

from sklearn.utils import validation
os.chdir(sys.path[0])
from sklearn.neural_network import MLPRegressor
# import the our class
from  nn_ik_demo import three_link_robot

import joblib   

#################################################################################
# @ Description: In this script , I will try to build a ANN to figure out the IK \
#                of the three link robot
#
### Notes: (1) there must be no duplicate data in the training data set
###        (2) the training data set should be scaled the value to [-1,1] interval
###        (3) the number of nerual from the input layer to the hidden layer is 3*100;
###            the number of nerual from the hidden layer to the output layer is 100*3
###        (4) the transfer function for the neurons in the hidden layer is the hyperbolic 
###            tangent function.(tanh)
###        (5) the transfer function for the neurons from the hidden layer to the output layer
###            is linear function.
###        (6) training algorithm is L-M algorithm
###        (7) 15% of the data set will be use to validation , 15% of the data set will be use to test
############################################################################################

class Neural_Network():
    """
        - Description: according to the Note point，build the Neural_Network for repeating
    """
    def __init__(self):
        self.__max_iter = 2500
        self.__neural_network_size = (100,100)
        self.__activation = "tanh"
        self.__solver = "adam"
        self.__mlpnn = MLPRegressor(
                             max_iter=self.__max_iter,
                             activation=self.__activation,
                             hidden_layer_sizes=self.__neural_network_size,
                             solver=self.__solver
                             )

    def get_data(self,filename):
        """
            - arguments: df is the DataFrame
        """
        df = pd.read_csv(filename)

        q1 = [];q2 = [];q3 = []
        X  = [];Y  = [];theta = []
        for row in df.itertuples():
            q1.append(row.q1);q2.append(row.q2);q3.append(row.q3)
            X.append(row.x_normalized);Y.append(row.y_normalized);theta.append(row.theta_normalized)
        
        train_set = []; target_set = []
        if (df.shape[0]):
            
            for i in range(df.shape[0]):
                train_set.append([X[i],Y[i],theta[i]])
                target_set.append([q1[i],q2[i],q3[i]])
            
            return train_set,target_set

        else:
            raise Exception("this data file missing some data")

    
    def train(self,train_set,target_set):

        regr = self.__mlpnn
        regr.fit(train_set,target_set)
        

        return regr
    

if __name__ == "__main__":

    # get the current path 
    filepath=os.getcwd()
    filename = filepath+'/data_normalized.csv'

    nn = Neural_Network()
    train_set , target_set = nn.get_data(filename)
    # 取 70 % 数据作为训练集
    # 取 15 % 数据作为验证集
    # 取 15 % 数据作为测试集
    length = len(train_set)
    nn_trained = nn.train(train_set[0:round(length*0.7)],target_set[0:round(length*0.7)])


    joblib.dump(nn_trained,'./nn_trained.pkl')
    test_set = train_set[round(length*0.7):(round(length*0.7)+round((length-length*0.7)/2))]
    test_target_set = target_set[round(length*0.7):(round(length*0.7)+round((length-length*0.7)/2))]

    validation_set = train_set[(round(length*0.7)+round((length-length*0.7)/2)):length]
    validation_target_set = target_set[(round(length*0.7)+round((length-length*0.7)/2)):length]

    print(nn_trained.score(test_set,test_target_set)) # 0.9507080591794769

    #################################################################################
    # If you change the solver to "adam" ,you can deannotation the code following
    #################################################################################
    df = pd.DataFrame(nn_trained.loss_curve_)
    print(df.to_string)
    df.plot()
    plt.show()
