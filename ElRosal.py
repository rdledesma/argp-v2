# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 14:06:53 2022

@author: Dario Ledesma
"""
from datetime import timedelta
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
import numpy as np

#Fn factor de correción 
def Fn(n):
    gamma = 2 * math.pi * (n - 1)/365
    fn = 1.000110 + 0.034221*math.cos(gamma) + 0.001280*math.sin(gamma) + 0.000719*math.cos(2*gamma) + 0.000077*math.sin(2*gamma)
    return fn

#Ecuación de tiempo
def getE(n):
    gamma = 2 * math.pi * (n-1)/365
    E = 229.18 * (0.000075+ 0.001868 * math.cos(gamma) - 0.032077*math.sin(gamma) - 0.014615*math.cos(2*gamma) - 0.04089*math.sin(2*gamma))
    return E


def delta(n):
    gamma = 2 * math.pi * (n- 1)/365
    delta = 0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma) - 0.006758 * math.cos(2*gamma) + 0.000907*math.sin(2*gamma) - 0.002697*math.cos(3*gamma)+ 0.00148*math.sin(3*gamma)
    return delta



#angulo horario
"""
longHusoHorario debe estar en grados
"""

def omega(Tutc,longLoc, longHusoHorario, n):
    return (math.pi/12)  * (Tutc - 12 + (longLoc - math.degrees(longHusoHorario))/15 + getE(n)/60)

#angulo cenital
"""
"""
def cosTitaZ(delta, lat, omega):
    #return math.sin(delta) * math.sin(lat) + math.cos(delta) * math.cos(lat)* math.cos(omega)
    return math.sin(delta) * math.sin(math.radians(lat)) + math.cos(delta) * math.cos(math.radians(lat))* math.cos(omega)


def sumar(n):
    gamma = 2 * math.pi * (math.cos(n/n)-1)/365
   
    return gamma

def Ts(Tutc, L0, Lutc, N):
    return Tutc + (L0-Lutc)/15 + getE(N)/60

def w(Ts):
    return math.pi*(Ts/12 - 1)


def maKY(lista, h):
    
    result = []
    for ctz in lista:
        tz = math.acos(ctz)
        
        try:
            if ctz == 0:
                result.append(0)
            else:
                result.append(math.exp(-0.0001184*h) / (ctz + 0.5057*math.pow((96.080 - tz), -1.634)) )
        except Exception:
            result.append(0)
        
    
    return result
   


lat = -24.4
longLoc = -65.7
longHusoHorario = -45
alt = 3360

data = pd.read_csv('./Data/ero.csv', sep=";", usecols=[0,2])

data[['Fecha d', 'Fecha h']] = data['Observation period'].str.split('/', 1, expand=True)
data['Fecha'] = pd.to_datetime(data['Fecha d']) + pd.DateOffset(minutes=7.5)
data['N'] = data['Fecha'].dt.day_of_year
data['E0'] = data['N'].apply(Fn)


data['Fecha'] = pd.to_datetime(data['Fecha'])
data['Hora reloj'] = data['Fecha'].dt.hour + data['Fecha'].dt.minute / 60
data['E'] =  data['N'].apply(lambda t: getE(t))
data['delta']  = data['N'].apply(lambda t: delta(t))

data['Ts'] = data.apply(lambda row: Ts(row['Hora reloj'],  longLoc , longHusoHorario ,row['N']), axis=1)
data['omega'] = data.apply(lambda row: w(row['Ts']), axis=1)

data['Fn'] = data['N'].apply(Fn)
data['Cos omega'] = data['omega'].apply(lambda t: math.cos(t))
data['Seno delta']  = data['delta'].apply(lambda t: math.sin(t))
data['Cos delta'] = data['delta'].apply(lambda t: math.cos(t))
data['CZ'] = data['Seno delta'] * math.sin(math.radians(lat)) + data['Cos delta']  * math.cos(math.radians(lat))* data['Cos omega']
#data['m']  = maKY(data['CZ'], alt)


data['m'] = 1/data['CZ']
data['TOA'] = np.where(data['CZ']>0, 1361 * data['Fn'] * data['CZ'],0)

data = data[data['Fecha'].dt.year==2010]







dias = []
Kt = []



dfGHIcc = pd.DataFrame()

# dfDia = data[data['N'] == 1 ]

def generateGHIcc(toas, ms, ktrp):
    ghis= []
    for i, item in enumerate(toas):
        try:
            result =  math.pow(math.pow( item * ktrp , ms[i]), 0.678)
            print(result)
            ghis.append(result) 
        except Exception:
            ghis.append(0) 
            print("ex")
    return ghis





ktrp = [i for i in np.arange(0.6, 0.7, 0.0001)]
testDias =[]
testRMSE = []
testKT = []
testError = []





for dia in [365]:
    dayDict = {}
    dfDia = data[data['N'] == dia]
    TOAs = dfDia['TOA']
    Mas = dfDia['m']
    for miKt in ktrp:
        
        
        
        GHIcc = generateGHIcc(dfDia['TOA'], dfDia['m'], miKt )
        
        #print(f" dia = {dia}, ktrp={GHIcc}")
        
        
        Error2 = dfDia['Clear sky GHI'] -GHIcc 
        dfGHIcc[""+str(miKt)] = GHIcc
        
        Error2 = Error2 ** 2
        RMSE = math.sqrt(sum(Error2)/96)
        dayDict.update({miKt : RMSE })
        
        testDias.append(dia)
        testRMSE.append(RMSE)
        testKT.append(miKt)
        #testError.append(dfDia['Clear sky GHI'] -GHIcc)
        
    valMin =  min(dayDict, key=dayDict.get)
    
    
    dias.append(dia)
    Kt.append(valMin)





df2 = pd.DataFrame()

df2['Dia'] = testDias
df2['RMSE'] = testRMSE
df2['Ktrp'] = testKT
#df2['Error'] = testError