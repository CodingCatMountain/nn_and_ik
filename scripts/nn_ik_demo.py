#!/usr/bin/env python3
#coding=utf-8
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plot
import os,sys
import csv

from sklearn import neural_network

###############################################
#  @Description: the DOI of the paper:10.1016/j.protcy.2013.12.451
#  @Date       : 2021.12.21
###############################################

### the global arguments:

# the range of the joint angles:
# q1 [0,pi]
# q2 [-pi,0]
# q3 [-pi/2,pi/2]

# x_e for the x of position of end_effector
# y_e for the y of position of end_effector
# theta_e is the orientation of end_effector

class three_link_robot():
    """
        - description: 针对论文中的三连杆机器人进行建模
        - methods    : forwordkinematic(q) q is a list, which include [q1,q2,q3]
        - memeber    : the length of links : l_1 l_2 l_3;              
    """
    def __init__(self):
        self.__l_1 = self.__l_2 = self.__l_3 = 2
    
    def forwordkinematic(self,q):
        """
            - description: calculate the forwardKinmatics of the three links robot
            - arguments  : the q is a list, which include [q_1,q_2,q_3]
            - return     : the list of [x,y,theta] ,theta is the orientation of the end_effector
        """

        x_e = self.__l_1*math.cos(q[0])+self.__l_2*math.cos(q[0]+q[1])+self.__l_3*math.cos(q[0]+q[1]+q[2])
        y_e = self.__l_1*math.sin(q[0])+self.__l_2*math.sin(q[0]+q[1])+self.__l_3*math.sin(q[0]+q[1]+q[2])
        theta_e = q[0] + q[1] + q[2]

        return [x_e,y_e,theta_e]

    def generateData(self):
        """
            - description: this methods is used to generate the data for Neural Network
            - the result : the data will sorted in the current document with the postfix of ***.csv
                           you can read the file with pandas
        """
        for i in range(0,3000):
            q_1 = np.random.rand()*math.pi              # the range of q_1 is [0,pi]
            q_2 = np.random.rand()*(-math.pi)           # the range of q_2 is [-pi,0]
            q_3 = np.random.rand()*(math.pi)-math.pi/2  # the range of q_3 is [-pi/2,pi/2]
            q = [q_1,q_2,q_3]
            location_e = self.forwordkinematic(q)

            # each values should be the object which could be iterated
            dict = {"q1":[q_1],\
                    "q2":[q_2],\
                    "q3":[q_3],\
                    "x":[location_e[0]],\
                    "y":[location_e[1]],\
                    "theta":[location_e[2]]
            }

            df = pd.DataFrame(dict)
            df.to_csv('data.csv',mode='a',header=False)

if __name__ == "__main__":
    robot = three_link_robot()
    robot.generateData()