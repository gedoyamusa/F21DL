#!/usr/bin/env python
# coding: utf-8

# In[4]:


import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is 
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import pandas as pd
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


# In[21]:


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


# In[22]:


df=pd.read_csv("diabetes.csv")


# In[23]:


# X = df.data[:, 2:] 
# y = df.target


# In[24]:


df.head()


# In[33]:


features = ['Preg','age','Plas','skin','test','mass','Pres','pedi']
X = df[features] 
y = df['class']


# In[34]:


dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

tree.plot_tree(dtree, feature_names=features)


# In[36]:


tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X, y)


# In[37]:


plot_tree(tree_clf);


# In[43]:


print(dtree.predict([[40, 10, 7, 1,3,3,2,5]])) #prediction of class


# In[ ]:




