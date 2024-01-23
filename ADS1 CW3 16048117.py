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
    df_transposed.columns = df_transposed.iloc[0]
    df_transposed = df_transposed.iloc[1:]
    df_transposed = df_transposed.apply(pd.to_numeric)
    df_transposed = df_transposed.dropna()

    return df_transposed, df_orig
