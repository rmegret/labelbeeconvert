#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 19:28:38 2017

@author: megret
"""

#%%
#%load_ext autoreload
#%autoreload 2

# In[]:
import sys
sys.path.append('/Users/megret/Documents/Research/BeeTracking/Soft/labelbee/python')


# In[]:
cd '/Users/megret/Documents/Research/BeeTracking/Soft/labelbeeconvert/data'

#%%
from labelbeeconvert import labelbee_convert as lb
import pandas as pd

#%%

df=lb.load_fileset('inputlist.csv')
df = df.query('FA!=True')
lb.plot_activities(df)

#%%
print(df['labels'].unique())
