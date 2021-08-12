#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor


# In[52]:


train = pd.read_csv("C:/Users/ABHISHEK/Desktop/iNeuron_Projects/Stores Sales Prediction/Dataset/Train.csv")


# In[53]:


train.head()


# In[54]:


train.info()


# In[55]:


train .isnull().sum()


# In[56]:


train.shape


# In[57]:


train [train ['Outlet_Size'].isnull()]


# In[58]:


train [train ['Item_Weight'].isnull()]


# In[59]:


# null values with heatmap
plt.figure(figsize=(16,9))
sns.heatmap(train .isnull())
plt.show()


# In[60]:


#persentage of null values with repect to total dataset
null_percent = train.isnull().sum() / train.shape[0] * 100
null_percent


# In[61]:


train.groupby("Item_Type")["Item_Weight"].mean()


# In[62]:


train.groupby("Item_Type")["Item_Weight"].mean()


# In[63]:


# Imputing null values with mean of Item Weights
mean = train.groupby('Item_Type')['Item_Weight'].mean()
for i in range(len(mean)):
    c1 = (train['Item_Type']==mean.index[i])&(train['Item_Weight'].isna()==True)
    train['Item_Weight'] = np.select([c1], [mean[i]], train['Item_Weight'])


# In[64]:


# Imputing null values with Mode
from statistics import mode
train['Outlet_Size'].fillna(mode(train['Outlet_Size']),inplace=True)


# In[65]:


# null values shows with heatmap
plt.figure(figsize=(16,9))
sns.heatmap(train .isnull())
plt.show()


# In[66]:


train .isnull().sum()


# In[67]:


# find out ow much unique types are present in Categorical variable
cols =["Item_Identifier","Item_Fat_Content","Item_Type","Outlet_Identifier","Outlet_Establishment_Year","Outlet_Size","Outlet_Location_Type","Outlet_Type"]
for i in cols:
    print(train[i].unique())


# # Univarient Analysis

# In[68]:


# Outlet_Type'
plt.figure(figsize=(8,5))
sns.countplot('Outlet_Type',data=train,palette='Set1')


# In[69]:


# Item_Fat_Content
plt.figure(figsize=(8,5))
sns.countplot('Item_Fat_Content',data=train,palette='Set1')


# In[70]:


# Outlet_Identifier
plt.figure(figsize=(8,5))
sns.countplot(train['Outlet_Identifier']);
plt.xticks(rotation = 45)
plt.show()


# In[71]:


# Item_Type 
plt.figure(figsize=(24,8))
sns.countplot('Item_Type',data=train,palette='twilight')


# In[95]:


plt.figure(figsize=(8,5))
sns.countplot('Outlet_Size',data=train,palette='Set2')


# In[72]:


# Outlet_Location_Type
plt.figure(figsize=(8,5))
sns.countplot('Outlet_Location_Type',data=train,palette='ocean')


# # Bi-varient Analysis

# In[73]:


plt.figure(figsize=(8,5))
sns.scatterplot(x='Item_Visibility',y='Item_Outlet_Sales',data=train);


# In[74]:


plt.figure(figsize=(12,5))
sns.barplot(x='Item_Type',y='Item_Outlet_Sales',data=train,palette='spring')
plt.xticks(rotation = 45)


# In[99]:


plt.figure(figsize=(12,5))
sns.boxplot(x='Outlet_Identifier',y='Item_Outlet_Sales',data =train);
plt.xticks(rotation = 90)
plt.show()


# In[75]:


plt.figure(figsize=(8,5))
sns.barplot(y='Item_Outlet_Sales',x='Outlet_Type',data=train);
plt.xticks(rotation = 0)
plt.show()


# In[76]:


plt.figure(figsize=(8,5))
sns.barplot(y='Item_Outlet_Sales',x='Outlet_Location_Type',data=train,palette='ocean');
plt.xticks(rotation = 0)
plt.show()


# In[77]:


plt.figure(figsize=(12, 9))
ax = sns.heatmap(data=train.corr(),cmap='coolwarm',  annot=True, linewidths=2)


# In[78]:


plt.figure(figsize=(8, 5))
sns.pairplot(data=train, hue="Item_Type")


# In[96]:


train.head()
train_x=train
train_x.head()


# In[97]:


# Dropping Unessesary column 
train_x = train_x.drop(columns=['Item_Identifier'])
train_x.head()


# In[98]:


#feature engineering for categorical varibles
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
var_mod = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Identifier','Outlet_Establishment_Year','Outlet_Location_Type', 'Outlet_Type']

for i in var_mod:
    train_x[i] = le.fit_transform(train_x[i])


# In[99]:


X = train_x.iloc[:,0:10]
X.head()


# In[100]:


Y=train_x.iloc[:,-1]
Y.head()


# In[101]:


print(X.shape,Y.shape)


# In[102]:


# Splitting the Dataset
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# In[103]:


print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)


# In[104]:


#metrics
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from sklearn.model_selection  import cross_val_score as CVS


# In[105]:


def cross_val(model_name,model,x,y,cv):
    
    scores = CVS(model, x, y, cv=cv)
    print(f'{model_name} Scores:')
    for i in scores:
        print(round(i,2))
    print(f'Average {model_name} score: {round(scores.mean(),4)}')


# # 1) Multiple Linear Regressor

# In[106]:


#ML model
from sklearn.linear_model import LinearRegression

#model
regressor_mlr = LinearRegression()

#fit
regressor_mlr.fit(x_train, y_train)

#predict
y_pred = regressor_mlr.predict(x_test)

#score variables
LR_MAE = round(MAE(y_test, y_pred),2)
LR_MSE = round(MSE(y_test, y_pred),2)
LR_R_2 = round(R2(y_test, y_pred),4)
LR_CS  = round(CVS(regressor_mlr, x, y, cv=5).mean(),4)

print(f" Mean Absolute Error: {LR_MAE}\n")
print(f" Mean Squared Error: {LR_MSE}\n")
print(f" R^2 Score: {LR_R_2}\n")
cross_val(regressor_mlr,LinearRegression(),x,y,5)


# In[49]:


Linear_Regression=pd.DataFrame({'y_test':y_test,'prediction':y_pred})
Linear_Regression.to_csv("Linear Regression.csv")


# # 2) Random Forest Regressor

# In[107]:


#ML model
from sklearn.ensemble import RandomForestRegressor

#model
regressor_rf = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4,random_state=101)

#fit
regressor_rf.fit(x_train, y_train)

#predict
y_pred = regressor_rf.predict(x_test)

#score variables
RFR_MAE = round(MAE(y_test, y_pred),2)
RFR_MSE = round(MSE(y_test, y_pred),2)
RFR_R_2 = round(R2(y_test, y_pred),4)
RFR_CS  = round(CVS(regressor_rf, x, y, cv=5).mean(),4)



print(f" Mean Absolute Error: {RFR_MAE}\n")
print(f" Mean Squared Error: {RFR_MSE}\n")
print(f" R^2 Score: {RFR_R_2}\n")
cross_val(regressor_rf,RandomForestRegressor(),x,y,5)


# In[ ]:


Random_Forest_Regressor=pd.DataFrame({'y_test':y_test,'prediction':y_pred})
Random_Forest_Regressor.to_csv("Random Forest Regressor.csv")  


# # 3) Lasso Regressor

# In[48]:


#ML model
from sklearn.linear_model import Lasso

#model
regressor_ls = Lasso(alpha = 0.05)
#fit
regressor_ls.fit(x_train,y_train)

#predict
y_pred = regressor_ls.predict(x_test)

#score variables
LS_MAE = round(MAE(y_test, y_pred),2)
LS_MSE = round(MSE(y_test, y_pred),2)
LS_R_2 = round(R2(y_test, y_pred),4)
LS_CS  = round(CVS(regressor_ls, x, y, cv=5).mean(),4)

print(f" Mean Absolute Error: {LS_MAE}\n")
print(f" Mean Squared Error: {LS_MSE}\n")
print(f" R^2 Score: {LS_R_2}\n")
cross_val(regressor_ls,Lasso(alpha = 0.05),x,y,5)


# # 4) XGBoost Regressor
# 

# In[50]:


#ML model
from xgboost import XGBRFRegressor

#model
regressor_xgb = XGBRFRegressor()

#fit
regressor_xgb.fit(x_train, y_train)

#predict
y_pred = regressor_xgb.predict(x_test)

#score variables
XGB_MAE = round(MAE(y_test, y_pred),2)
XGB_MSE = round(MSE(y_test, y_pred),2)
XGB_R_2 = round(R2(y_test, y_pred),4)
XGB_CS  = round(CVS(regressor_xgb, x, y, cv=5).mean(),4)

print(f" Mean Absolute Error: {XGB_MAE}\n")
print(f" Mean Squared Error: {XGB_MSE}\n")
print(f" R^2 Score: {XGB_R_2}\n")
cross_val(regressor_xgb,XGBRFRegressor(alpha = 0.05),x,y,5)


# # 5) Ridge Regressor

# In[51]:


#ML model
from sklearn.linear_model import Ridge

#model
regressor_rd = Ridge(normalize=True)
#fit
regressor_rd.fit(x_train,y_train)

#predict
y_pred = regressor_ls.predict(x_test)

#score variables
RD_MAE = round(MAE(y_test, y_pred),2)
RD_MSE = round(MSE(y_test, y_pred),2)
RD_R_2 = round(R2(y_test, y_pred),4)
RD_CS  = round(CVS(regressor_rd, x, y, cv=5).mean(),4)

print(f" Mean Absolute Error: {RD_MAE}\n")
print(f" Mean Squared Error: {RD_MSE}\n")
print(f" R^2 Score: {RD_R_2}\n")
cross_val(regressor_rd,Ridge(normalize=True),x,y,5)


# In[ ]:




