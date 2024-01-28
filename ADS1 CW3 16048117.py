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
import cluster_tools as ct


#functions

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


def colour_map_countries(my_df, indicator):
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
    plt.title( indicator + " correlation by country 1990-2020", fontsize=20, 
              weight='bold', va='top')
    plt.xticks(ticks=[0, 1, 2, 3, 4], labels=annotations, fontsize = 20)
    plt.yticks(ticks=[0, 1, 2, 3, 4], labels=annotations, fontsize = 20)
    
    plt.show()
    plt.tight_layout() 
    return

def merge_GDP_and_CO2():
    """
    Merging the data for my 2 chosen indicators GDP and CO2 for 2020
    
    """
    
    GDP = pd.read_csv("GDP_per_capita.csv")
    print(GDP.describe())
     
    # read CO2
    CO2 = pd.read_csv("CO2_emmisions.csv")
    print(CO2.describe())
    
    # create a new dataframe using CO2
    CO2_and_GDP_2020 = CO2[["Country Name", "2020"]].copy()
    
    # rename the data column
    CO2_and_GDP_2020 = CO2_and_GDP_2020.rename(columns={"2020": "CO2"})
    
    # copy a column from GDP
    CO2_and_GDP_2020["GDP"] = GDP["2020"]
    CO2_and_GDP_2020 = CO2_and_GDP_2020.dropna()
    print(CO2_and_GDP_2020)
    return  CO2_and_GDP_2020

def clusterplot ():
    """
    This function creates a cluster plot for my merged data for comparing
    CO2 emmissions with GDP per country world wide
    
    """    
    
    
    df_GDP_and_CO2 = merge_GDP_and_CO2()
    
    # extract columns for fitting
    df_fit = df_GDP_and_CO2[["CO2", "GDP"]].copy()
    
    # normalise dataframe and inspect result
    df_fit, df_min, df_max = ct.scaler(df_fit)
    print(df_fit.describe())
    print("n   score")
    
    # loop over trial numbers of clusters calculating the silhouette
    for ic in range(2, 7):
        # set up kmeans and fit
        kmeans = cluster.KMeans(n_clusters=ic)
        kmeans.fit(df_fit)     
    
        # extract labels and calculate silhoutte score
        labels = kmeans.labels_
        print (ic, skmet.silhouette_score(df_fit, labels))
    
    
    # Plot it on the original scale
    nc = 4
    
    kmeans = cluster.KMeans(n_clusters=nc)
    kmeans.fit(df_fit)     
    
    # extract labels and cluster centres
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    
    plt.figure(figsize=(6.0, 6.0))
    # scatter plot with colours selected using the cluster numbers
    # now using the original dataframe
    plt.scatter(df_GDP_and_CO2["CO2"], df_GDP_and_CO2["GDP"], c=labels, cmap="tab10")
    # colour map Accent selected to increase contrast between colours
    
    # rescale and show cluster centres
    scen = ct.backscale(cen, df_min, df_max)
    xc = scen[:,0]
    yc = scen[:,1]
    plt.scatter(xc, yc, c="k", marker="d", s=80)
    
    plt.xlabel("CO2 emmissions", fontsize=20)
    plt.ylabel("GDP per capita", fontsize=20)
    plt.title("4 cluster plot GDP and CO2", fontsize=20, weight='bold', va='top')
    plt.show()
    return

#main program
#read CO2 emmissions file
df_CO2_transposed, df_CO2_orig = reading_my_data("CO2_emmisions.csv")

#select the 5 countries I'm interested in CO2
CO2_5_countries = df_CO2_transposed.iloc[0:, 0:6]

#call colour map function for correlation of CO2 production between countries
colour_map_countries(CO2_5_countries, "CO2 emmissions")
print(CO2_5_countries.describe())

#read CO2 emmissions file
df_GDP_transposed, df_GDP_orig = reading_my_data("GDP_per_capita.csv")

#select the 5 countries I'm interested in GDP
GDP_5_countries = df_GDP_transposed.iloc[0:, 0:6] 
print(GDP_5_countries.describe())

#call colour map function for correlation of GDP production between countries
colour_map_countries(GDP_5_countries, "GDP per capita")

#call my clusterplot
clusterplot()




