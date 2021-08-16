#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("Companies-profit.csv")
df.head()


# In[3]:


from sklearn import linear_model


# In[4]:


x = df[['R&D Spend', 'Administration', 'Marketing Spend']]
y = df['Profit']


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state =10)


# In[7]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[8]:


model.fit(x_train, y_train)


# In[9]:


model.score(x_test, y_test)


# In[10]:


import pickle
pickle.dump(model, open('profit.pkl', 'wb'))

