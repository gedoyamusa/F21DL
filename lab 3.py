#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn


# In[2]:


data=pd.read_csv('ds_salaries.csv')


# In[3]:


data.tail()


# In[4]:


data.corr()  # prints the correlation coefficient between every pair of attributes


# In[5]:


import seaborn as sns

sns.pairplot(data, kind="reg")  # plots scatter plots for every pair of attributes and histograms along the diagonal
plt.show()


# In[6]:


fig,ax = plt.subplots(figsize=(10, 10))   
sns.heatmap(data.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma") # the color intensity is based on 
plt.show()


# In[ ]:





# In[7]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[8]:


data['experience_level']=le.fit_transform(data['experience_level'])
data['employment_type']=le.fit_transform(data['employment_type'])
data['job_title']=le.fit_transform(data['job_title'])
data['salary_currency']=le.fit_transform(data['salary_currency'])
data['employee_residence']=le.fit_transform(data['employee_residence'])
data['company_location']=le.fit_transform(data['company_location'])
data['company_size']=le.fit_transform(data['company_size'])



# In[9]:


data.head()


# In[31]:


data.drop(columns=data.columns[0],axis=1,inplace=True)
data.head()


# In[32]:


corr_matrix=data.corr()  # prints the correlation coefficient between every pair of attributes
corr_matrix


# In[33]:


import seaborn as sns

sns.pairplot(data, kind="reg")  # plots scatter plots for every pair of attributes and histograms along the diagonal
plt.show()


# In[34]:


acorr_matrix['company_location'].sort_values(ascending=False)


# In[44]:


dataset1=data[["employee_residence","salary_currency","salary_in_usd","experience_level", "remote_ratio","company_size","salary","employment_type", "job_title"  ]]
dataset2=data[["employee_residence","salary_currency","salary_in_usd","experience_level", "remote_ratio"]]
dataset3=data[["employee_residence","salary_currency","salary_in_usd"]]


# In[35]:


corr_matrix['job_title'].sort_values(ascending=False)


# In[14]:


corr_matrix['employment_type'].sort_values(ascending=False)


# In[15]:


# if r > 0.9 (r < = -0.9) - Strong correlation
# else if  r >= 0.65 (r <= -0.65) - moderate correlation
# else  if  r >= 0.2 (r <= -0.2) -a weak correlation


# In[ ]:





# In[16]:


#work_year             
#nnamed: 0            0.167025
#remote_ratio          0.132122
#job_title  


# In[17]:


#data1 = data1.drop('remote_ratio', axis=1)
#data1 = data.drop(['job_title','remote_ratio','work_year'], axis=1)#'Unnamed'
data.drop(columns=data.columns[0],axis=1,inplace=True)
data.head()


# In[ ]:


dataset1=data[[""]]
dataset2=data[[""]]
dataset3=data[[""]]


# In[19]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from PIL import Image 


# In[ ]:





# # UNIVARIATE ANALYSIS

# In[20]:


dataset = data['salary']
len(dataset)


# In[21]:


dataset.isnull().sum()


# In[22]:


# The following code plots a histrogram using the matplotlib package.
# The bins argument creates class intervals. In this case we are creating 50 such intervals
plt.hist(dataset, bins=50)


# In the above histogram, the first array is the frequency in each class and the second array contains the edges of the class intervals. These arrays can be assigned to a variable and used for further analysis.
# 

# In[23]:


sns.distplot(dataset) # plots a frequency polygon superimposed on a histogram using the seaborn package.
# seaborn automatically creates class intervals. The number of bins can also be manually set.


# In[24]:


sns.distplot(data, hist=False) # adding an argument to plot only frequency polygon


# In[25]:


sns.violinplot(dataset) # plots a violin plt using the seaborn package.


# In[26]:


plt.figure(figsize=(20,10)) # makes the plot wider
plt.hist(dataset, color='g') # plots a simple histogram
plt.axvline(dataset.mean(), color='m', linewidth=1)
plt.axvline(dataset.median(), color='b', linestyle='dashed', linewidth=1)
plt.axvline(dataset.mode()[0], color='w', linestyle='dashed', linewidth=1)


# If we notice univariate analysis does not properly work with less dataset

# # Spliting Data
# 

# In[28]:


import sklearn
from sklearn.model_selection import train_test_split
y=data['company_location']
#X = data1[data1.columns.difference(['company_location'])]
X=data[['employee_residence','salary_currency']]
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.25,random_state=42)


# # KNN Classifier

# In[40]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)


# In[41]:


#accuracy
from sklearn import metrics
print("accuracy:",metrics.accuracy_score(y_test, y_pred))


# # RANDOM FOREST CLASSIFIER

# In[42]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
forest_clf= RandomForestClassifier(max_depth=7,random_state=0)
forest_clf.fit(X_train,y_train)
res_pred=forest_clf.predict(X_train)
accuracy_score(y_train, res_pred)


# In[ ]:




