# -*- coding: utf-8 -*-
"""
Created on Mon May  9 17:08:29 2022

@author: Dario Ledesma
"""
import math
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
class Geo:
    
    def __init__(self, freq, lat, long, gmt ):
        self.gmt = gmt
        self.lat = lat
        self.long = long
        self.altura = 31
        self.df = pd.DataFrame()
        self.df['Fecha'] =  pd.date_range(start='1/1/2010 00:00:00', end='31/12/2010 23:59:00', freq= freq+' min')
        self.df['Fecha'] =  self.df['Fecha'] + pd.DateOffset(minutes=7.5)
        self.df['Fecha'] =  self.df['Fecha']
        self.df['N'] = self.df['Fecha'].dt.day_of_year
        self.df['E'] = self.df['N'].apply(self.getE)
        self.df['HR'] = self.df['Fecha'].dt.hour + (self.df['Fecha'].dt.minute) /60 + (self.df['Fecha'].dt.second)/3600
        self.df['HS'] = self.getHS()
        self.df['delta rad'] = self.df['N'].apply(self.delta)
        self.df['delta'] = self.df['delta rad'].apply(math.degrees)
        self.df['w'] = 15 * (12 - self.df['HS'])    
        self.df['w rad'] = self.df['w'].apply(math.radians)
        self.df['CTZ'] =  self.df.apply(lambda r: self.getCTZ(r['delta rad'], self.lat, r['w rad']), axis=1)
        self.df['TZ'] = self.df['CTZ'].apply(math.acos)
        self.df['E0'] = self.df['N'].apply(self.getE0)
        self.df['TOA'] = self.df.apply(lambda r: self.TOA(r['E0'], r['CTZ']), axis=1 )
        self.df['Ma'] = self.generateMa()
        self.df['Ma2'] = self.getMA(self.df['CTZ'])
        self.df['Mak'] = self.MaK()
        self.ktrp = 0.7 + 1.6391 * 10**-3 * self.altura ** 0.5500 
        self.df['GHIargp'] = self.generateGHIargp(self.df)
        self.df['GHIcc'] = self.df.apply(lambda r: self.generateGHIcc(r['TOA'], r['Mak'], self.ktrp), axis=1)
        
        
        
        
    #Ecuación de tiempo
    def getE(self,n):
        gamma = 2 * math.pi * (n-1)/365
        E = 229.18 * (0.000075+ 0.001868 * math.cos(gamma) - 0.032077*math.sin(gamma) - 0.014615*math.cos(2*gamma) - 0.04089*math.sin(2*gamma))
        return E
    
    
    
    def getHS(self):
        A = 1
        if self.gmt<0:
            A = -1
        return self.df['HR'] + (4 * ((A * 15 * self.gmt)- (A*self.long))+ self.df['E'])/60
    
    def delta(self, N):
        gamma = 2 * math.pi * (N- 1)/365
        delta = 0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma) - 0.006758 * math.cos(2*gamma) + 0.000907*math.sin(2*gamma) - 0.002697*math.cos(3*gamma)+ 0.00148*math.sin(3*gamma)
        return delta
    
    def getCTZ(self, delta, lat, omega):
        latR = math.radians(self.lat)    
        return (math.cos(latR) * math.cos(delta)* math.cos(omega)) + (math.sin(latR)*math.sin(delta))
        #return math.sin(delta) * math.sin(math.radians(lat)) + math.cos(delta) * math.cos(math.radians(lat))* math.cos(omega)

    def getE0(self, N):
        return 1+0.033* math.cos(2* math.pi * N /365)
    
    def TOA(self, E0, CTZ):
        if CTZ<0:
            return 0
        else:
            return 1361 * E0 * CTZ
        
        
    def getMA(self,CTZs):
        result = []
        for ctz in CTZs:
            try:
                valor1 = 1.002432 * ctz**2 + 0.148386*ctz + 0.0096
                valor2 = ctz**3 + 0.149864*ctz**2 + 0.0102963*ctz + 0.000303978
                result.append(valor1/valor2)
            except Exception:
                result.append(0)
        return result
    
    
    
        
    
    def MaK(self):
        presion = 101355* (288.15/(288.15 - 0.0065 * self.altura)) ** -5.255877
        
        CTZ = self.df['CTZ']
        TZ = self.df['TZ']
        
        Amc = []
        for i, val in enumerate(CTZ):
            Amk = 1/ (val + 0.15*(93.885 - TZ[i])**-1.253)
            Amc.append( Amk * (presion/101355))
            
        return Amc
    
    
    def generateMa(self):
        
        cosTZ = self.df['CTZ'].tolist()
        tz = self.df['TZ'].tolist()
           
        
        
        
        presion =  math.pow(288.15/(288.15 - 0.0065 * 1150), -5.255877);
        results = []
        
        for i, val in enumerate(cosTZ):
            
            try:
                calc = 1/(val + 0.15 * math.pow((93.885-  tz[i]), -1.253));
                results.append(calc * presion)
            except Exception:
                results.append(0)        
        return results
    
    def generateGHIcc(self, TOA, AM, ktrp):
        try:
            return TOA * math.pow( ktrp ,math.pow(AM, 0.678))
        except Exception: 
            return 0
    
    def generateGHIargp(self, data):
        GHI = data['TOA'].tolist()
        AM = data['Mak'].tolist()
    
    
    
        result = []
        for i, val in enumerate(GHI):
            try:
                result.append(GHI[i] * math.pow( self.ktrp ,math.pow(AM[i], 0.678)))
            except Exception:
                result.append(0)
        return result
        

        


import promedio
def generateGHIcc(TOA, AM, ktrp):
    try:
        return TOA * math.pow( ktrp ,math.pow(AM, 0.678))
    except Exception: 
        return 0

df = Geo(freq='15', lat=-34.90, long=-56.20, gmt=-3).df

salta  = promedio.prom('./Data/montevideo.csv')


salta['TOA'] = salta['TOA'] *4
salta['Clear sky GHI'] = salta['Clear sky GHI'] *4


salta['TOA'] = salta['TOA'].shift(-12)
salta['Clear sky GHI'] = salta['Clear sky GHI'].shift(-12)
salta.fillna(0, inplace=True)
df['GHImc'] = salta['Clear sky GHI'].copy()



df2 = df.copy()
df = df[df['CTZ']>=0]


df = df[['TOA','N','Mak','GHImc']]

ktrs = np.arange(0.5, 0.9999, 0.0001)
minKtErrors = []
arrayDia = []
arrayKtr = []
arrayMSE = []
for dia in range(1,366):
    myDf = df[df['N']==dia]
    dayDict = {}
    for ktr in ktrs:
        GHIcc = myDf.apply(lambda r: generateGHIcc(r['TOA'], r['Mak'], ktr), axis=1)
        mse = mean_squared_error(myDf['GHImc'], GHIcc)
        arrayDia.append(dia)
        arrayKtr.append(ktr)
        arrayMSE.append(mse)
        dayDict.update({ktr: mse })
    
    valMin =  min(dayDict, key=dayDict.get)
    minKtErrors.append(valMin)
    
    
dfMin = pd.DataFrame()

dfMin['Monte'] = minKtErrors

dfMin.to_csv('monteMin.csv',sep=";", index=False)

