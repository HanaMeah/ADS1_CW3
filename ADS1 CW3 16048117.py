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


def logistic_function(t, n0, g, t0):
    
    """
    This function calculates the logistic function with the scale factor n0 
    and growth rate g
    
    """
    
    logistic_f = n0 / (1 + np.exp(-g*(t - t0)))
    return logistic_f


def colour_map_countries(my_df):
    """
    This function plots the correlation colour map of CO2 production 
    between my selected countries when called with my selected dataframe in my_df
    
    """
    plt.figure(1)
    

    corr = my_df.corr(numeric_only=True)
    print(corr.round(3))
    plt.figure(figsize=[10, 8])
    
    plt.imshow(corr)
    plt.colorbar()
    annotations = CO2_5_countries.columns[1:6] # extract relevant headers
    plt.title("CO2 emmissions correlation by country 1990-2020", fontsize=20, 
              weight='bold', va='top')
    plt.xticks(ticks=[0, 1, 2, 3, 4], labels=annotations, fontsize = 20)
    plt.yticks(ticks=[0, 1, 2, 3, 4], labels=annotations, fontsize = 20)
    
    plt.show()
    plt.tight_layout() 
    return



df_CO2_transposed, df_CO2_orig = reading_my_data("CO2_emmisions.csv")
CO2_5_countries = df_CO2_transposed.iloc[0:, 0:6]

#call colour map function for correlation of CO2 production between countries
colour_map_countries(CO2_5_countries)

print(CO2_5_countries.describe())




