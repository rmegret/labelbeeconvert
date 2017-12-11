#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 19:28:38 2017

@author: megret
"""


#%%
%load_ext autoreload
%autoreload 2

# In[]:
import sys
sys.path.append('/Users/megret/Documents/Research/BeeTracking/Soft/labelbeeconvert')

#%%
from labelbeeconvert import labelbee_convert as lb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# In[]:
cd '/Users/megret/Documents/Research/BeeTracking/Soft/labelbeeconvert/data'

#%%

evts=lb.load_fileset('inputlist_Gurabo.csv')
tags=lb.load_tags_fileset('tags_files.csv')
#df = df.query('FA!=True')
#lb.plot_activities(df)

#%%
def compute_tagsize(p):
    return np.sqrt(((p[0][0]-p[2][0])**2+(p[0][1]-p[2][1])**2
                    + (p[1][0]-p[3][0])**2+(p[1][1]-p[3][1])**2)/2)

tags[['tagsize']]=tags[['p']].applymap(lambda p: compute_tagsize(p))

#%%
def compute_tagminedge(p):
    return np.sqrt(
            np.min([
                    (p[0][0]-p[1][0])**2+(p[0][1]-p[1][1])**2,
                    (p[1][0]-p[2][0])**2+(p[1][1]-p[2][1])**2,
                    (p[2][0]-p[3][0])**2+(p[2][1]-p[3][1])**2,
                    (p[3][0]-p[0][0])**2+(p[3][1]-p[0][1])**2
                    ])
                    )

tags[['tagminedge']]=tags[['p']].applymap(lambda p: compute_tagminedge(p))

#%%
def compute_area(p):
    return np.abs( (p[1][0]-p[0][0])*(p[3][1]-p[0][1])
             - (p[1][1]-p[0][1])*(p[3][0]-p[0][0])
            ) + np.abs(
                (p[1][0]-p[2][0])*(p[3][1]-p[2][1])
                - (p[1][1]-p[2][1])*(p[3][0]-p[2][0])
                    )
,

tags[['area']]=tags[['p']].applymap(lambda p: compute_area(p))

#%%

def compute_x(c):
    return c[0]
def compute_y(c):
    return c[1]

tags[['x']]=tags[['c']].applymap(lambda c: compute_x(c))
tags[['y']]=tags[['c']].applymap(lambda c: compute_y(c))

#%%
print(df['labels'].unique())


#%%



# In[]:
cd '/Users/megret/Documents/Research/BeeTracking/Soft/labelbee/data/Gurabo'

# In[]:
tags = lb.load_tags_df('Tags-C02_170624090000-raw.json')
evts = lb.load_tracks_df('workshop/C02_170624090000-Tracks-Grace.json')

# In[]:
plt.plot(evts['frame'],evts['id'],'o')
plt.plot(tags['frame'],tags['id'],'.')

# In[]:
e2=evts.copy()
e2=e2.merge(tags,'left', on=['video','frame','id'])
e3=e2.loc[e2['dm'].notnull(),['video','frame','id','dm','falsealarm','wrongid','c','p']]
#e3=e2.loc[e2['dm'].notnull(),:]

#%%
m=0

L=[]
for i in range(-m,m+1):
    tmp = evts.copy()
    tmp[['frame']] = tmp[['frame']]+i
    L.append(tmp)

E = pd.concat(L)
E = E.merge(tags,'left', on=['video','frame','id'])
E = E.loc[E['dm'].notnull(),:]

#E = E.query('dm>20 and tagsize>35')
E = E.query('dm>20')

#%%
plt.figure()

#prop = 'dm'
prop = 'tagminedge'

H,edges=np.histogram(E[prop],100)
plt.plot(edges[:-1],H,label='all')

H_FA,edges2=np.histogram(E.query('falsealarm')[prop],edges)
plt.plot(edges[:-1],H_FA,label='falsealarm')

H_WI,edges2=np.histogram(E.query('wrongid')[prop],edges)
plt.plot(edges[:-1],H_WI,label='wrongid')

H2,edges2=np.histogram(E.query('not wrongid and not falsealarm')[prop],edges)
plt.plot(edges[:-1],H2,label='correct')

plt.legend()

#%%
plt.figure()
H,edges=np.histogram(df.loc[df['dm'].notnull(),'dm'],100,)
plt.plot(edges[:-1],H,label='all')

#%%
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
fig=plt.figure()

cls=E['wrongid']+2*(E['falsealarm'] & ~E['wrongid'])

classes=['OK','WId','FA']
cmap = colors.ListedColormap(['green', 'blue','red'])
centers=range(cmap.N)
bounds=np.array(range(cmap.N+1))-0.5
norm = colors.BoundaryNorm(bounds, cmap.N)

ax = fig.add_subplot(111, projection='3d')
handle=ax.scatter(E['tagsize'],E['area'],E['tagminedge'],s=1,
                  c=cls, cmap=cmap, norm=norm)
cbar=plt.colorbar(mappable=handle)
cbar.set_ticks(centers)
cbar.set_ticklabels(classes)
