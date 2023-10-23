#!/usr/bin/env python
# coding: utf-8

# In[128]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
plt.style.use('ggplot')

import random
import warnings
import itertools
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[129]:


#reading data from the csv file 
#and storing x values as month and y values which are monthly average price index of coffee as sale
#K is number of data we got
X= np.genfromtxt("CoffeePrice.csv",delimiter =",")
month= X[:,0].astype(int)
sale = X[:,1]
K=X.shape[0]
plt.figure(figsize = (16,8))
plt.plot(month,sale,"b")
plt.xlabel("month")
plt.ylabel("sale")
plt.title("Actual Sales")
plt.legend(['Actual Sales'])
plt.show


# In[130]:


# plotting the 12 month seasonal figure 
#in order to investigating the seasonality
plt.figure(figsize = (16,8))
plt.xlabel("month")
plt.ylabel("sale")
plt.title("Actual Sales")
plt.legend(['Actual Sales'])
plt.show
for i in range(30):
    plt.plot(month[0:12],sale[(i+1)*12:(i+2)*12],label='$y = {i}x + {i}$'.format(i=i))
plt.show


# In[131]:


#implementation of naive one month forecast
#shifting the sale values and month values one month 
#for 1991-2020 period 
est1_sale= sale[11:K-14]
est1_month= month[12:K-13]
#residual calculations for 1991-2020 period 
error1 = sale[12:K-13]-est1_sale
mae1 = np.mean(np.abs(error1))
MAPE1 = ((np.sum(np.abs(error1)/sale[12:K-13]))*100)/len(sale[12:K-13])
mse1=np.mean(error1**2)
rmse1 = np.sqrt(mse1)
#plotting the figure
plt.figure(figsize = (16, 8))
plt.plot(month,sale, 'b')
plt.plot(est1_month,est1_sale ,'r')
plt.xlabel("month")
plt.ylabel("sales")
plt.legend(['actual sales','est1 sales','error1'])
plt.text(-20,120,'MAE1={}'.format(mae1),fontsize=10)
plt.text(-20,110,'MAPE1={}'.format(MAPE1),fontsize=10)
plt.text(-20,100,'RMSE1={}'.format(rmse1),fontsize=10)
plt.show()


# In[132]:


#implementation of 5-period moving average 
#for 1991-2020 period 
#shifted np arrays that are used for forecasting
est2_sales=(sale[7:K-18]+sale[8:K-17]+sale[9:K-16]+ sale[10:K-15]+ sale[11:K-14])/5
est2_month=month[12:K-13]
#residual calculations for 1991-2020 period 
error2=sale[12:K-13]-est2_sales
mae2=np.mean(np.abs(error2))
mse2=np.mean(error2**2)
rmse2=np.sqrt(mse2)
MAPE2 = ((np.sum(np.abs(error2)/sale[12:K-13]))*100)/len(sale[12:K-13])
#plotting the figure
plt.figure(figsize = (16, 8))
plt.plot(month,sale, 'b')
plt.plot(est2_month,est2_sales, 'r')

plt.xlabel("month")
plt.ylabel("sales")
plt.title("Moving Average")
plt.legend(['actual sales','est2 sales','error2'])
plt.text(-20,120,'MAE2={}'.format(mae2),fontsize=10)
plt.text(-20,110,'MAPE2={}'.format(MAPE2),fontsize=10)
plt.text(-20,100,'RMSE2={}'.format(rmse2),fontsize=10)
plt.show()


# In[133]:


#calculation of %95 prediction interval with RMSE for 5-period moving average 
#for one month ahead
nextMonthPrediction= (sale[K-17]+sale[K-16]+sale[K-15]+ sale[K-14]+ sale[K-13])/5
PredictionIntervalUpper2= nextMonthPrediction +rmse2*1.96
PredictionIntervalLower2= nextMonthPrediction -rmse2*1.96
print('Lower prediction Interval' , PredictionIntervalLower2)
print('Upper prediction Interval',PredictionIntervalUpper2)


# In[134]:


# implementation of exhaustive search for the best smoothing constant α (alpha)
yhates=np.zeros(K)
yhates[0]=sale[0]
bestrmse=10000
for j in range(1,100):
    alpha=1/j
    for i in range(1,K):
        yhates[i] =alpha*sale[i-1]+(1-alpha)*(yhates[i-1])
    errores=sale[1:K]- yhates[1:K]
    msees=np.mean(np.square(errores))
    rmsees= np.sqrt(msees)           
    if bestrmse> rmsees:
        bestrmse=rmsees
        bestalpha = alpha
        
print('Optimal alpha = ', bestalpha)
print('Optimal RMSE = ', bestrmse)


# In[135]:


# implementation of exponential smoothing to forecast the one-month ahead price
#with using optimal alpha

alpha= bestalpha
est3_sale=np.zeros(K)
est3_sale[0]= sale[0]
for k in range(1,K):
    est3_sale[k]= alpha*sale[k-1]+ (1-alpha)*est3_sale[k-1]
est3_month=month[12:K-13]
#calculation of residuals for 1991-2020 period 
error3= sale[12:K-13]-est3_sale[12:K-13]
mae3 = np.mean(np.abs(error3))
mse3=np.mean(error3**2)
MAPE3 = ((np.sum(np.abs(error3)/sale[12:K-13]))*100)/len(sale[12:K-13])
rmse3 = np.sqrt(mse3)
#plotting the figure for 1991-2020 period 
plt.figure(figsize = (16, 8))
plt.plot(month,sale, 'b')
plt.plot(est3_month,est3_sale[12:K-13] ,'r')
plt.text(-20,120,'MAE3={}'.format(mae3),fontsize=10)
plt.text(-20,110,'MAPE3={}'.format(MAPE3),fontsize=10)
plt.text(-20,100,'RMSE3={}'.format(rmse3),fontsize=10)

plt.show()


# In[136]:


#calculation of %95 prediction interval with RMSE for exponential smoothing
#for one month ahead
nextMonthPrediction=est3_sale[K-12]
PredictionIntervalUpper3= nextMonthPrediction +rmse3*1.96
PredictionIntervalLower3= nextMonthPrediction -rmse3*1.96
print('Lower prediction Interval' , PredictionIntervalLower3)
print('Upper prediction Interval',PredictionIntervalUpper3)


# In[137]:


#implementation of naive forecast that
#includes trend
est4_sales=(sale[2:K-1] + (sale[1:K-2] -sale[0:K-3]))
est4_month=month[3:K]
#calculation of residuals for 1991-2020 period 
error4=sale[12:K-13]-est4_sales[12:K-13]
mae4=np.mean(np.abs(error4))
mse4=np.mean(error4**2)
rmse4=np.sqrt(mse4)
MAPE4 = ((np.sum(np.abs(error4)/sale[12:K-13]))*100)/len(sale[12:K-13])
#plotting the figure for 1991-2020 period 
plt.figure(figsize = (16, 8))
plt.plot(month,sale, 'b')
plt.plot(est4_month[12:K-13],est4_sales[12:K-13], 'r')
plt.xlabel("month")
plt.ylabel("sales")
plt.legend(['actual sales','est2 sales','error2'])
plt.text(-20,120,'MAE4={}'.format(mae4),fontsize=10)
plt.text(-20,90,'MAPE4={}'.format(MAPE4),fontsize=10)
plt.text(-20,100,'RMSE4={}'.format(rmse4),fontsize=10)
plt.show()


# In[138]:


#calculation of %95 prediction interval with RMSE for naive forecast that
#that includes trend
#for one month ahead
nextMonthPrediction=est4_sales[K-12]
PredictionIntervalUpper4= nextMonthPrediction +rmse4*1.96
PredictionIntervalLower4= nextMonthPrediction -rmse4*1.96
print('Lower prediction Interval' , PredictionIntervalLower4)
print('Upper prediction Interval',PredictionIntervalUpper4)


# In[139]:


# implementation of the exponentially smoothed version of naive forecast that
#includes trend with alpha = 0.7 and beta = 0.2
alpha = 0.7
beta = 0.2
est5_sale = np.zeros(K)
#in order to keep track of zt and update zthat
#i created 2 np array
# which are zthat and zt.
#both of them is calculated as 0 for the first iteraration

# first index of zt is zero
zt= np.array([0] + [sale[i]-sale[i-1] for i in range(1,K)])
est5_sale[0]= sale[0]
zthat = np.zeros(len(zt))
#first index of zthat is zero
zthat[0] = zt[0]
for k in range(1,K):
    est5_sale[k]= alpha*sale[k-1]  + (1-alpha)*est5_sale[k-1]+beta*zt[k-1]+(1-beta)*zthat[k-1]
    #updated zthat after each iteration 
    zthat[k] = est5_sale[k]-est5_sale[k-1]

#calculation of residuals for 1991-2020 period 
est5_month=month[12:K-13]
error5= sale[12:K-13]-est5_sale[12:K-13]
mae5 = np.mean(np.abs(error5))
mse5=np.mean(error5**2)
MAPE5 = ((np.sum(np.abs(error5)/sale[12:K-13]))*100)/len(sale[12:K-13])
rmse5 = np.sqrt(mse5)

#plotting the figure for 1991-2020 period 
plt.figure(figsize = (16, 8))
plt.plot(month,sale, 'b')
plt.plot(est5_month,est5_sale[12:K-13] ,'r')
plt.text(-20,120,'MAE5={}'.format(mae5),fontsize=10)
plt.text(-20,90,'MAPE5={}'.format(MAPE5),fontsize=10)
plt.text(-20,100,'RMSE5={}'.format(rmse5),fontsize=10)

plt.show()


# In[140]:


#calculation of %95 prediction interval with RMSE for exponentially smoothed version of naive forecast that
#includes trend with alpha = 0.7 and beta = 0.2
nextMonthPrediction=est5_sale[K-12]
PredictionIntervalUpper5= nextMonthPrediction +rmse5*1.96
PredictionIntervalLower5= nextMonthPrediction -rmse5*1.96
print('Lower prediction Interval' , PredictionIntervalLower5)
print('Upper prediction Interval',PredictionIntervalUpper5)


# In[141]:


#implementation of searching for 
# optimal values of α and β that minimize the RMSE for sixmonth ahead forecasts
#for exponentially smoothed version of naive forecast that
#includes trend 
#for years 1991 through 2020
bestrmse=10000
K = len(sale)
slope= np.array([0] + [sale[i]-sale[i-1] for i in range(1,len(sale))])
for j in range(1,100):
    alpha=1/j
    for m in range(1,100):
        beta=1/m 
        est6_sale = np.zeros(K)
        est6_sale[0]= sale[0]
        slopehat = np.zeros(len(slope))
        slopehat[0] = slope[0] 
        for k in range(1,K-6):
            est6_sale[k+5]= alpha*sale[k-1]  + (1-alpha)*est6_sale[k-1]+ 7*(beta*slope[k-1]+ (1-beta)*slopehat[k-1])
            slopehat[k] = est6_sale[k]-est6_sale[k-1]
        errores=sale[12:K-13]- est6_sale[12:K-13]
        msees=np.mean(np.square(errores))
        rmsees= np.sqrt(msees)           
        if bestrmse> rmsees:
            bestrmse=rmsees
            bestalpha=alpha
            bestbeta = beta
print('Optimal alpha = ', bestalpha)
print('Optimal Beta = ', bestbeta)


# In[142]:


# implementation of sixmonth ahead forecasts
#for exponentially smoothed version of naive forecast that
#includes trend with optimal alpha and beta
#for years 1991 through 2020
alpha = bestalpha
beta = bestbeta
est6_sale = np.zeros(K)
zt= np.array([0] + [sale[i]-sale[i-1] for i in range(1,K)])
est6_sale[0]= sale[0]
zthat = np.zeros(len(zt))
zthat[0] = zt[0]
#i used same logic with one month ahead version of exponentially smoothed version of naive forecast that
#includes trend, but it includes weight 7 and makes predictions for 
#6 month ahead
for k in range(1,K-6):
    est6_sale[k+5 ]= alpha*sale[k-1]  + (1-alpha)*est6_sale[k-1]+  7*(beta*zt[k-1]+(1-beta)*zthat[k-1])
    zthat[k] = est6_sale[k]-est6_sale[k-1]

#calculation of residuals for 1991-2020 period 
est6_month=month[12:K-13]
error6= sale[12:K-13]-est6_sale[12:K-13]
mae6 = np.mean(np.abs(error6))
mse6=np.mean(error6**2)
MAPE6 = ((np.sum(np.abs(error6)/sale[12:K-13]))*100)/len(sale[12:K-13])
rmse6 = np.sqrt(mse6)

#plotting the figure for 1991-2020 period 
plt.figure(figsize = (16, 8))
plt.plot(month,sale, 'b')
plt.plot(est6_month,est6_sale[12:-13] ,'r')
plt.text(-20,120,'MAE5={}'.format(mae6),fontsize=10)
plt.text(-20,90,'MAPE5={}'.format(MAPE6),fontsize=10)
plt.text(-20,100,'RMSE5={}'.format(rmse6),fontsize=10)

plt.show()


# In[143]:


# Generate an AR (1) process 
#with c = 50, ϕ1 = 0.6 and ϵt are normally distributed with mean zero
#and standard deviation 20.
#with with 500 realizations
c=50; sigma=20; phi1=0.6; # parameters
eps = np.random.normal(0, sigma, 501); #  normally distributed vector of error terms
y_ar=[0]*501;
y_ar[0]=c; 
for i in range(1,501):
    y_ar[i] =c + phi1*y_ar[i-1]  + eps[i] # auto-regression on the previous observation


# In[144]:


time_axis = np.linspace(0,501,501) # generate the x-axis values
start = 99 # start from observation 100 to eliminate the effects of initialization
plt.figure(figsize = (16, 8))
plt.plot(time_axis[start:500],y_ar[start:500], 'b') 
plt.xlabel("x")
plt.ylabel("y ")
plt.legend(["y "])
plt.show()


# In[145]:


# The auto-correlation function plot start from observation 100 to eliminate the effects of initialization
sm.graphics.tsa.plot_acf(y_ar[99:500], lags=10);


# In[146]:


#Implementation a naive forecast on the data I generated
month = np.array([i for i in range(0,501)])
y_ar=np.array(y_ar)
yhat= y_ar[98:499]
ymonth= month[99:500]
#calculation of RMSE
error1 = y_ar[99:500] - yhat
mae1 = np.mean(np.abs(error1))
MAPE1 = ((np.sum(np.abs(error1)/y_ar[99:500]))*100) /len(y_ar[99:500])
mse1=np.mean(error1**2)
rmse1 = np.sqrt(mse1)

plt.figure(figsize = (16, 8))
plt.plot(month,y_ar, 'b')
plt.plot(ymonth,yhat ,'r')

plt.xlabel("x")
plt.ylabel("y")
plt.title("AR")

plt.text(0,60,'MAE={}'.format(mae1),fontsize=10)
plt.text(0,50,'MAPE={}'.format(MAPE1),fontsize=10)
plt.text(0,40,'RMSE={}'.format(rmse1),fontsize=10)
plt.show()


# In[147]:


# The auto-correlation function plot of sale
sm.graphics.tsa.plot_acf(sale, lags=20);


# In[148]:


# The partial-auto-correlation function plot of the sale
sm.graphics.tsa.plot_pacf(sale, lags=20);


# In[149]:


#Differencing the data or detrending the data 
y_t1= sale[0:K-1]
y= sale[1:K]
u= y-y_t1
est1_month= month[12:K-13]
error1 = sale[12:K-13]-est1_sale


# In[150]:


# The auto-correlation function plot of the differenced sale data
sm.graphics.tsa.plot_acf(u, lags=20);


# In[152]:


# The partial-auto-correlation function plot of the differenced sale data
sm.graphics.tsa.plot_pacf(u, lags=20);


# In[ ]:




