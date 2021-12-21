#!/usr/bin/env python3
#coding=utf-8
import pandas as pd
import os,sys
os.chdir(sys.path[0])

df  = pd.read_csv(r'./data_normalized.csv')
print(df.to_string())
print(df.shape)
#print(df["x_normalized"])
#print(df["y_normalized"])
#print(df["theta_normalized"])