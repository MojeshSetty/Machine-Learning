#!/usr/bin/env python
# coding: utf-8

# In[1]:


#NaiveBayes project (Weather Prediction)
#Required Modules
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB


# In[2]:


df = pd.read_csv("Weather Data.csv")
df


# In[3]:


#Encoding the strings to Numericals
outlook_at=LabelEncoder()
Temp_at=LabelEncoder()
Hum_at=LabelEncoder()
win_at=LabelEncoder()


# In[4]:


#Dropping the target variable and make it is as newframe
inputs=df.drop('Play',axis='columns')
target=df['Play']
target


# In[5]:


#Creating the new dataframe
inputs['outlook_n']= outlook_at.fit_transform(inputs['Outlook'])
inputs['Temp_n']= outlook_at.fit_transform(inputs['Temp'])
inputs['Hum_n']= outlook_at.fit_transform(inputs['Humidity'])
inputs['win_n']= outlook_at.fit_transform(inputs['Windy'])
inputs


# In[6]:


#Dropping the string values
inputs_n=inputs.drop(['Outlook','Temp','Humidity','Windy'],axis='columns')
inputs_n


# In[8]:


#Applying the Gaussian naivebayes
classifier = GaussianNB()
classifier.fit(inputs_n,target)
#GaussianNB()


# In[11]:


# accuracy 
classifier.score(inputs_n,target)


# In[12]:


#Prediction
classifier.predict([[0,0,0,1]])

