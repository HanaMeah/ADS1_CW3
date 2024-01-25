# -*- coding: utf-8 -*-

"""
Spyder Editor

MSc Data Science
Applied Data Science 1
Assignment 2 - 30 %
Hana Meah
16048117

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt


def reading_my_data(datafilename):
    """

    General function to read all of my data csv files which gives the output 
    of my original dataframe and my transposed data frame and cleans the 
    transposed dataframe

    """
    df_orig = pd.read_csv(datafilename)
    columns = df_orig.columns[1:]

    df_orig[columns] = df_orig[columns].apply(pd.to_numeric)

    # Transposing my dataframe
    df_transposed = df_orig.transpose()
    
    #clean transposed data
    df_transposed.columns = df_transposed.iloc[0]
    df_transposed = df_transposed.iloc[1:]
    df_transposed = df_transposed.apply(pd.to_numeric)
    df_transposed = df_transposed.dropna()
    
    # reset index and insert a column for 'Year'
    df_transposed.reset_index(drop=True, inplace=True)
    df_transposed.insert(0, 'Year',
                      ['1990', '1991', '1992','1993', '1994', '1995', '1996',
                        '1997', '1998', '1999', '2000', '2001', '2002', '2003',
                        '2004', '2005', '2006', '2007', '2008', '2009', '2010',
                        '2011', '2012', '2013', '2014', '2015', '2016', '2017',
                        '2018', '2019', '2020'])

    return df_transposed, df_orig

df_CO2_transposed, df_CO2_orig = reading_my_data("CO2_emmisions.csv")

df_CO2_transposed.head(10)


