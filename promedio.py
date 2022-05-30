# -*- coding: utf-8 -*-
"""
Created on Mon May  9 15:58:05 2022

@author: Dario Ledesma
"""

import pandas as pd
import numpy as np

def prom(archivo):
    df = pd.read_csv(archivo, sep=";", usecols=[1,2])
    df['Fecha'] = pd.date_range(start='1/1/2010 00:00:00', end='31/12/2020 23:59:00', freq='15 min')
    biciestos = df.index[(df['Fecha'].dt.year.isin([2012,2016,2020])) & (df['Fecha'].dt.day_of_year == 60)].tolist()
    df.drop(biciestos, inplace=True)
    
    df['N'] = df['Fecha'].dt.day_of_year
    df['N'] = np.where((df['Fecha'].dt.year.isin([2012,2016,2020])) & (df['N'] > 59), df['N']-1, df['N'])

    prom = df.groupby([df['N'],  df['Fecha'].dt.hour   , df['Fecha'].dt.minute]).mean()
    prom.rename_axis(['N','H','M'], inplace=True)
    return prom.reset_index()
    




