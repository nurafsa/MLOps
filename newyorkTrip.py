#!/usr/bin/env python
# coding: utf-8

# In[74]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[75]:


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[76]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[77]:


def read_dataframe(filename):
    df = pd.read_parquet(filename)
    
    df['Duration']= df['dropOff_datetime']-df['pickup_datetime']
    df['Duration']=df['Duration'].apply(lambda td: td.total_seconds()/60)
    
    df=df[(df.Duration >=1)& (df.Duration <=60)]
    
    df['PUlocationID'] = df['PUlocationID'].replace(np.nan,-1)
    df['DOlocationID'] = df['DOlocationID'].replace(np.nan,-1)
    
    categorical=['PUlocationID','DOlocationID']
    df[categorical].astype(str)
    
    return df    


# In[78]:


df_train=read_dataframe('fhv_tripdata_2021-01.parquet')
df_val=read_dataframe('fhv_tripdata_2021-02.parquet')


# In[79]:


categorical=['PUlocationID','DOlocationID']

dv=DictVectorizer()
train_dict=df_train[categorical].to_dict(orient='records')
X_train=dv.fit_transform(train_dict)

val_dict=df_val[categorical].to_dict(orient='records')
X_val=dv.fit_transform(val_dict)


# In[80]:


target='Duration'
Y_train=df_train[target].values
Y_val=df_val[target].values


# In[81]:


model = LinearRegression() 
model.fit(X_train, Y_train)

y_pred = model.predict(X_train)
y_val_pred=model.predict(X_val)

RMSE_train = mean_squared_error(Y_train, y_pred, squared=False)
print(RMSE_train)
RMSE_val = mean_squared_error(Y_val, y_val_pred, squared=False)
print(RMSE_val)


# # Following is practice code

# In[41]:


dfJan.head()


# In[42]:


dfJan.shape


# In[43]:


dfJan.dtypes


# In[64]:


dfJan['Duration']= dfJan['dropOff_datetime']-dfJan['pickup_datetime']
dfJan['Duration']=dfJan['Duration'].apply(lambda td: td.total_seconds()/60)
dfJan.head()


# In[45]:


average= dfJan['Duration'].mean()
print(average)


# In[46]:


sns.distplot(dfJan["Duration"])


# In[47]:


dfJan['Duration'].describe()


# In[48]:


dfJan.isnull().sum()


# In[49]:


958267/1154112


# In[65]:


dfJan=dfJan[(dfJan.Duration >=1)& (dfJan.Duration <=60)]


# In[66]:


dfJan['PUlocationID'] = dfJan['PUlocationID'].replace(np.nan,-1)
dfJan['DOlocationID'] = dfJan['DOlocationID'].replace(np.nan,-1)
dfJan


# In[67]:


categorical=['PUlocationID','DOlocationID']


# In[52]:


dfJan[categorical].astype(str).dtypes


# In[68]:


train_dict=dfJan[categorical].to_dict(orient='records')


# In[69]:


dv=DictVectorizer()
X_train=dv.fit_transform(train_dict)
X_train


# In[70]:


target='Duration'
Y_train=dfJan[target].values


# In[71]:


model = LinearRegression() 
model.fit(X_train, Y_train)


# In[72]:


y_pred = model.predict(X_train)


# In[73]:


RMSE = mean_squared_error(Y_train, y_pred, squared=False)
print(RMSE)


# In[59]:


sns.distplot(y_pred, label='prediction')
sns.distplot(Y_train, label='actual')
plt.legend()

