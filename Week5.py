# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 17:20:47 2023

@author: adfw980
"""

#Exercise 1

# Download the data, open the file, load it into a Pandas DataFrame

f = open('C:/Users/adfw980/Downloads/train.csv')

import csv

import pandas as pd

hprice_train = pd.DataFrame(csv.reader(f))

#Plot the distributions of the columns that indicate Living Area and Price:
    #GrLivArea and SalePrice
    
import matplotlib.pyplot as plt

    #Histogram and Boxplot for Livign Area
    
    hprice_train = hprice_train.iloc[1:]
    
    hprice_train[46] = hprice_train[46].astype(float)
    
    plt.hist(hprice_train[46])
    
    plt.boxplot(hprice_train[46])
    
    #As we can see from the histogram/boxplots the living Area is Negativelyu Skewed
    #Also, it presents a considerable number of upper outliers
    
    #Histograms and boxplots for the Price column:
        
    hprice_train[80] = hprice_train[80].astype(float)
    
    plt.hist(hprice_train[80])

    plt.boxplot(hprice_train[80])
    
    #Similarly for the boxplots and histograms here
    
    #We will make a scatterplot without removing the outliers first
    
    plt.scatter(hprice_train[80], hprice_train[46])
    
    #Here we can see a considerable number of outliers, it is necessary that we eliminate the outliers from both columns
    #However, we can still observe a Trend line 
    
    #We will eliminate the outliers by using the IQR and make a new dataframe without outliers for LivingSize
    
    Q1 = hprice_train[46].quantile(0.25)

    Q3 = hprice_train[46].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound= Q1 - 1.5 * IQR

    upper_bound = Q3 + 1.5 * IQR
    
    no_outliers_Size = hprice_train[(hprice_train[46] >= lower_bound) & (hprice_train[46] <= upper_bound)]
    
    # We will do the same for Price:
        
        Q1 = hprice_train[80].quantile(0.25)

        Q3 = hprice_train[80].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound= Q1 - 1.5 * IQR

        upper_bound = Q3 + 1.5 * IQR

        no_outliers_Price = hprice_train[(hprice_train[80] >= lower_bound) & (hprice_train[80] <= upper_bound)]
    
    #Nowe we will compute a single DataFrame
    
    no_outliers_Price[46]=no_outliers_Size[46]
    
    no_outliers_df = no_outliers_Price
   
    plt.scatter(no_outliers_df[46], no_outliers_df[80])
    
    #We can observe a better represenation of the trend line as we eliminated the outliers
    
    #4 We will build a regression between Living Size and Price
    
  from sklearn import datasets, linear_model
  
  from sklearn.metrics import mean_squared_error, r2_score
  
  #We will do a simple Regression where Living Size (X variable) Predicts House Price(YVariable)
  
  #First we will train the model, we will split the data intro train and test groups
  

  no_outliers_df_X_train = no_outliers_df[46][:700]
  no_outliers_df_X_test = no_outliers_df[46][-699:]
    
  no_outliers_df_Y_train = no_outliers_df[80][:700]
  no_outliers_df_Y_test = no_outliers_df[80][-699:]

#We will create the linear regression model as an object, then we will train the model on the train data with regr.fit

regr = linear_model.LinearRegression()

#We will transform the columns in DataFrames

no_outliers_df_X_test = pd.DataFrame(no_outliers_df_X_test)

no_outliers_df_X_train = pd.DataFrame(no_outliers_df_X_train)

no_outliers_df_Y_train = pd.DataFrame(no_outliers_df_Y_train)

no_outliers_df_Y_test = pd.DataFrame(no_outliers_df_Y_test)

#Filling the NAns with the Mean value of the column

no_outliers_df_X_test.fillna(no_outliers_df_X_test.mean(), inplace=True)

no_outliers_df_X_train.fillna(no_outliers_df_X_train.mean(), inplace=True)

no_outliers_df_Y_test.fillna(no_outliers_df_Y_test.mean(), inplace=True)

no_outliers_df_Y_train.fillna(no_outliers_df_Y_train.mean(), inplace=True)

#Calculate the regression model

regr.fit(no_outliers_df_X_train, no_outliers_df_Y_train)

#We will calculate the prediciton of the Y value following the Regression model we trained

no_outliers_df_Y_pred = regr.predict(no_outliers_df_X_test)

#We will now calculate the coefficients and make the scatterplot

print("Coefficients: \n", regr.coef_)

#Mean Squared Error

print("Mean squared error: %.2f" % mean_squared_error (no_outliers_df_Y_test, no_outliers_df_Y_pred))

print("Coefficient of determination: %.2f" % r2_score(no_outliers_df_Y_test, no_outliers_df_Y_pred))

plt.scatter(no_outliers_df_X_test, no_outliers_df_Y_test, color="black")

plt.scatter(no_outliers_df_X_test, no_outliers_df_Y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())
plt.show()

#Part5 

