# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:05:15 2024

@author: 刘66
"""

import pandas as pd
import os 
os.getcwd()
os.chdir('D:\\Project\\Domain Adaptation\\TransferCDR\\data\\raw_dat\\CCLE') 
df = pd.read_csv('CCLE_expression.csv', index_col=0)
mean = df.mean()
std = df.std()
z_score = (df - mean) / std

z_score.to_csv('zscore.csv')
