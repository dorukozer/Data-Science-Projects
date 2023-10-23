#!/usr/bin/env python
# coding: utf-8

# In[33]:


get_ipython().run_line_magic('matplotlib', 'inline')
#Doruk ÖZER 70192
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.metrics import confusion_matrix, classification_report


# In[34]:


df = pd.read_csv('Toyota_Sales.csv', index_col=0) # let's data into dataframe df
training_set= df[:96]# seperating data into training set
test_set= df[96:]# seperating data into test set
df.head() 


# In[35]:


# Running a linear regression with 11 dummies (no dummy at May) with t,tsq and tcube
lm = sm.OLS.from_formula('Sales ~ t + tsq + tcube + Jan + Feb + Mar + Apr + Jun + Jul + Aug + Sep + Oct + Nov + Dec ', training_set) 
result = lm.fit() # results of the OLS regression


# In[36]:


print(result.summary())
# the many of the dummies are statistically insignificant based on the observations at the P values
# t, and tsq are statically significant
# R^2 value is low for training set. It states that the model can predict 0.38 of the predictions  
# also monthly dummies such as March April September November October are statistically insignificant. 
# correlation between prediction and actual values are quite bad


# In[37]:


MSE1= np.mean(np.square(result.resid)) # we can retrieve the residuals: e_t= y_t-yhat_t
RMSE1 = np.sqrt(MSE1) # and compute the usual error measures
MAPE1 = (np.mean(abs(np.array(result.resid))/np.array(training_set.iloc[:,1])))
print('MAPE of Training Set = ', MAPE1)
print('MSE of Training Set =', MSE1)
print('RMSE of Training Set =', RMSE1)


# In[38]:


time_axis = np.linspace(1,500,499)
plt.figure(figsize = (16, 8))
plt.plot(time_axis[0:96],np.array(training_set.iloc[:,1]), 'r')
plt.plot(time_axis[0:96], result.get_prediction().predicted_mean, 'b')
plt.xlabel("month")
plt.ylabel("Sales")
plt.title("Sales")
plt.legend(['Sales'])
plt.show()


# In[39]:


# using our model estimated at trainging set on the test set 
result2=result.predict(test_set)
# the residuals on the test set
resids_test1=test_set.iloc[:,1] - result2
#MSE MAPE RMSE on the test set.
MAPE_test = (np.mean(abs(np.array(resids_test1))/np.array(test_set.iloc[:,1])))
print('MAPE of Test Set = ', MAPE_test)
mse_test1= np.mean(np.square(resids_test1))
rmse_test1 = np.sqrt(mse_test1)
print('MSE Test  =', mse_test1)
print('RMSE Test =', rmse_test1 )


# In[40]:


# Running a linear regression with 11 dummies (no dummy at May) with t,tsq and tcube
lm = sm.OLS.from_formula('Sales ~ t + Jan + Dec ', training_set) 
result = lm.fit() # results of the OLS regression


# In[41]:


print(result.summary())

#R^2 value is nearly the same, and  dummies at Jan and Dec is statistically significant also the trend term t has P value
# as 0.307 can be counted as statistically significant. 
# Reduced model shows us that reduced model has nearly performs as well as previous full-model. This shows that, the predictors of the previous
# model contains unnecessary predictors.


# In[42]:


MSE1= np.mean(np.square(result.resid)) # retrieving the residiuals
RMSE1 = np.sqrt(MSE1) # computation of RMSE
MAPE1 = (np.mean(abs(np.array(result.resid))/np.array(training_set.iloc[:,1]))) # computation of MAPE
print('MAPE of Training Set = ', MAPE1)
print('MSE of Training Set =', MSE1)
print('RMSE of Training Set =', RMSE1)


# In[43]:


#printing the predictions
time_axis = np.linspace(1,500,499)
plt.figure(figsize = (16, 8))
plt.plot(time_axis[0:96],np.array(training_set.iloc[:,1]), 'r')
plt.plot(time_axis[0:96], result.get_prediction().predicted_mean, 'b')
plt.xlabel("month")
plt.ylabel("Sales")
plt.title("Sales")
plt.legend(['Sales'])
plt.show()


# In[44]:


# using our model estimated at trainging set on the test set 
result2=result.predict(test_set)
# the residuals on the test set
resids_test1=test_set.iloc[:,1] - result2
#MSE MAPE RMSE on the test set.
MAPE_test = (np.mean(abs(np.array(resids_test1))/np.array(test_set.iloc[:,1])))
print('MAPE of Test Set = ', MAPE_test)
mse_test1= np.mean(np.square(resids_test1))
rmse_test1 = np.sqrt(mse_test1)
print('MSE Test  =', mse_test1)
print('RMSE Test =', rmse_test1 )

# Better results at test set since the overfitting issue is discarded by reducing our model parameters
# other parameters in the previous model should have caused some overfitting.
# In addition, previous model were nearly as succesful as the reduced model on the training set altough it was more complex
# and containing more parameters.


# In[45]:


#read the data to a dataframe
df1 = pd.read_csv('Toyota_Sales_Up.csv', index_col=0, parse_dates=True)
training_set= df1[:96]# seperating data into training set
test_set= df1[96:168]# seperating data into test set

training_set.head()


# In[46]:


# since we have a predictor at lag 3, we can start making predictions at month 5 
training_set1=training_set[5:]


# In[47]:


# Defining model to fit with Lag1 , Lag2 ,Lag3 and monthly indicators: Jan and Dec
formula = 'Up ~ Lag1+Lag2+Lag3+ Jan +Dec'


# In[48]:


model1 = smf.glm(formula = formula, data=training_set1, family=sm.families.Binomial())
result1 = model1.fit()
result1.summary()
# results shows us that indicators at Dec and Jan are insignificant since their P values are 0.999
# the lag1 lag2 lag3 are statistically significant
# that means ups and downs are not realted with the monthly indicators.


# In[49]:


#Predicting the probabilities 
predictions1 = result1.predict() 
# assigning 1/2 as treshold to classify the results as 1 or 0 
# if the probabilities are higher than 1/2 than it would be classified as class 1 
pthreshold = 0.5
predictions1_class = [ 1 if x > pthreshold else 0 for x in predictions1]
print(predictions1_class)


# In[50]:


# based on confusion matrix
# total error rate calculated: (15+10)/92
# we were able to predict 42 out of the 52 up periods as up
# the model predicted 25 out of 40 down periods
print(confusion_matrix(training_set1.iloc[:,1], predictions1_class))


# In[51]:


#report
print(classification_report(training_set1.iloc[:,1], predictions1_class))


# In[52]:


# AUC score 
# it is close to 1 which is great
from sklearn.metrics import roc_auc_score
roc_auc_score(training_set1.iloc[:,1], predictions1_class)


# In[53]:


# checking the error performance of the model on the test set
# error rate is 21/73
# model catches 36 of 39 ups 
predictions_test=result1.predict(test_set)
threshold=0.5
predictions2_class_test = [ 1 if x > threshold else 0 for x in predictions_test] 
print(confusion_matrix(np.array(test_set.iloc[:,1]), predictions2_class_test))


# In[54]:


print(classification_report(np.array(test_set.iloc[:,1]), predictions2_class_test))


# In[55]:


print(roc_auc_score(np.array(test_set.iloc[:,1]), predictions2_class_test))
# AUC is lower in test set and is a little far from 1 


# In[56]:


# Defining model to fit with Lag1 , Lag2 , and monthly indicator: Dec 
formula = 'Up ~ Lag1+Lag2+Dec'


# In[57]:


model1 = smf.glm(formula = formula, data=training_set1, family=sm.families.Binomial())
result1 = model1.fit()
result1.summary()
# results shows us that indicator at Dec  are insignificant since their P values are 0.999
# the lag1 lag2  are statistically significant
# that means ups and downs are not realted with the monthly indicators.


# In[58]:


#Predicting the probabilities  with reduced model
predictions1 = result1.predict() 
# assigning 1/2 as treshold to classify the results as 1 or 0 
# if the probabilities are higher than 1/2 than it would be classified as class 1 
pthreshold = 0.5
predictions1_class = [ 1 if x > pthreshold else 0 for x in predictions1]
print(predictions1_class)


# In[59]:


# based on confusion matrix
# total error rate calculated: (12+10)/92 which is worse than the full model
# we were able to predict 42 out of the 52 up periods as up
# there is an Improvment in this model that we catch more down periods compared to full model
# 28 out of 40 of downs are predicted correctly !!!!!!!!!
print(confusion_matrix(training_set1.iloc[:,1], predictions1_class))


# In[60]:


#report of reduced model on training set
print(classification_report(training_set1.iloc[:,1], predictions1_class))


# In[61]:


# AUC score 
# it is close to 1 which is great and it is better than the full model
from sklearn.metrics import roc_auc_score
roc_auc_score(training_set1.iloc[:,1], predictions1_class)


# In[62]:


# checking the error performance of the reduced model on the test set
# error rate is 26/73 better than full model !!!
# model catches 33 of 39 ups  full model is better !!!!!
# model catches 14 of 34 of downs worse than previous model !!
predictions_test=result1.predict(test_set)
threshold=0.5
predictions2_class_test = [ 1 if x > threshold else 0 for x in predictions_test] 
print(confusion_matrix(np.array(test_set.iloc[:,1]), predictions2_class_test))


# In[63]:


print(classification_report(np.array(test_set.iloc[:,1]), predictions2_class_test))


# In[64]:


print(roc_auc_score(np.array(test_set.iloc[:,1]), predictions2_class_test))
# AUC is lower in test set and lower than full-model in terms of test set and is a little far from 1 


# In[ ]:





# In[ ]:




