# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:40:24 2022

@author: Dario Ledesma
"""
from sklearn.metrics import mean_squared_error
import pandas as pd 
import numpy as np
import math

df = pd.read_csv('./salta-MC-ARGP.csv', sep=";")




ktrs = np.arange(0.55, 0.65, 0.0001)
minKtErrors = []
arrayDia = []
arrayKtr = []
arrayMSE = []
for dia in range(1,10):
    myDf = df[df['N']==dia]
    dayDict = {}
    for ktr in ktrs:
        GHIcc = myDf.apply(lambda row:  math.pow( row['TOA'] * ktr ,row['Mak']), axis=1)     
        
        
        
        mse = mean_squared_error(myDf['GHImc'], GHIcc)
        arrayDia.append(dia)
        arrayKtr.append(ktr)
        arrayMSE.append(mse)
        dayDict.update({ktr: mse })
    
    valMin =  min(dayDict, key=dayDict.get)
    minKtErrors.append(valMin)
