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
sys.path.append('/Users/megret/Documents/Research/BeeTracking/Soft/apriltag/swatbotics-apriltag/python/')

#%%
from labelbeeconvert import labelbee_convert as lb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# In[]:
cd '/Users/megret/Documents/Research/BeeTracking/Soft/labelbeeconvert/data'


#%%

evts=lb.load_fileset('inputlist_Gurabo.csv')

#%%
tags=lb.load_tags_fileset('tags_files.csv')

#%%

evts_val = evts.query('falsealarm!=True & wrongid!=True')
evts_val.index = range(evts_val.shape[0])
evts_val.reindex()
lb.plot_activities(evts_val);
plt.tight_layout();

#%%
#def compute_wrongid(labels):
#    return 'wrongid' in labels
#
#T1[['wrongid']]=T1[['labels']].applymap(lambda labels: compute_wrongid(labels))

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

sys.path.append('/Users/megret/Documents/Research/BeeTracking/Soft/apriltag/swatbotics-apriltag/python')

#%%

def plot_tags(tags_df, ax=None, labels=None):
    """Plots apriltag tags on matplotlib axis. tags in DataFrame format"""

    if (ax is not None):
        plt.sca(ax)
            
    for i,tag in tags_df.iterrows():
        p=np.array(tag.p)
        
        if (labels is None):
            label = str(tag.id)
        else:
            label = str(labels[i])
        
        infotext = "{tag.id} H{tag.hamming} dm{tag.dm:.0f}".format(**locals())
        
        #print(c[:,1])
        pp,=plt.plot(p[:,0],p[:,1],'r-')
        plt.plot(p[:,0],p[:,1],'o',markeredgecolor=pp.get_color(), markerfacecolor="None")
        plt.text(np.mean(p[:,0]),np.min(p[:,1])-5,label,fontsize=12,horizontalalignment='center', verticalalignment='bottom', color=pp.get_color())
        plt.text(np.mean(p[:,0]),np.max(p[:,1])+5,infotext,fontsize=9,horizontalalignment='center', verticalalignment='top', color=pp.get_color())

#%%
        
def extract_tag_image(tag, rgb, rotate=True, S=None, n=9, pixsize=None):
    
    #frame = tag.frame
    
    #vidcap.set(cv2.CAP_PROP_POS_MSEC,1000.0/fps*(frame-1))
    #status,orig = vidcap.read();
    
    #gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
    #rgb = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)

    if (rotate):
        if (S is None and pixsize is None):
            S = n*pixsize
            s = pixsize
        else if (S is None):
            pixsize = 10
            S = n*pixsize
            s = pixsize
            
        pts_src = np.array(tag.p)
        #pts_dst = np.array([[0,0],[S,0],[S,S],[0,S]])
        pts_dst = np.array([[s,s],[S-s,s],[S-s,S-s],[s,S-s]])
        size = (S,S)
    else:
        if (S is None):
            S = 90
        R = S/2
        C = np.array(tag.p).mean(0)
        pts_src = C + np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*R
        pts_dst = np.array([[0,0],[S,0],[S,S],[0,S]])
        size = (S,S)
    
    h, status = cv2.findHomography(pts_src, pts_dst)
    tag_img = cv2.warpPerspective(rgb, h, size)
    
    return tag_img

def extract_tag_image_vid(tag, vidcap, rotate=True):
    
    frame = tag.frame
    
    vidcap.set(cv2.CAP_PROP_POS_MSEC,1000.0/fps*(frame-1))
    status,orig = vidcap.read();
    
    #gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
    rgb = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)

    tag_img = extract_tag_image(tag, rgb, rotate=rotate, S=90)
    
    return tag_img

#%%

import cv2
#%%

folder='/Users/megret/Documents/Research/BeeTracking/Soft/labelbee-current/data/Gurabo'
video_in = folder+'/C02_170622100000.mp4'
fps = 20

vidcap = cv2.VideoCapture(video_in)


#%%

# =============================================================================
#[ 31, 195, 200, 266, 289, 296, 303, 312, 359, 387, 394, 409, 410, 444, 459,
#  498, 502, 508, 516, 517, 529, 541, 567, 607, 611, 638, 648, 673, 694, 731,
#  762, 820, 855, 856, 869, 874, 886, 912, 921, 924, 927, 928, 929, 930, 935,
#  950, 964, 975, 976, 980, 989, 990, 1039, 1095, 1099, 1112, 1133, 1136, 1147,
#  1190, 1214, 1230, 1278, 1296, 1361, 1380, 1415, 1420, 1438, 1467, 1484, 1487,
#  1503, 1512, 1555, 1562, 1565, 1573, 1600, 1601, 1602, 1609, 1621, 1658, 1689,
#  1691, 1698, 1701, 1716, 1739, 1769, 1786, 1797, 1802, 1819, 1824, 1830, 1835,
#  1837, 1847, 1849, 1854, 1870, 1878, 1883, 1914, 1921, 1927, 1951, 1959, 1974,
#  1978, 2048, 2057, 2061, 2087, 2104, 2122, 2130, 2137, 2167, 2184, 2199, 2236,
#  2244, 2245, 2295, 2298, 2337, 2338, 2356, 2360, 2362, 2376, 2392, 2393, 2394,
#  2398, 2415, 2416, 2433, 2439, 2440, 2456, 2460, 2464, 2479, 2481, 2487, 2505,
#  2522, 2563, 2569, 2579, 2583, 2587, 2596, 2608, 2626, 2627, 2633, 2640, 2641,
#  2649, 2655, 2666, 2675, 2690, 2700, 2725, 2732, 2737, 2741, 2749, 2750, 2752,
#  2774, 2782, 2805, 2808, 2835, 2836, 2844, 2856, 2873, 2884, 2900, 2905, 2923,
#  2927, 2947, 2952, 2957, 2962, 2978, 2980, 2987, 2993, 2995, 3006, 3007]
# =============================================================================

T1 = tags.query('video=="C02_170622100000"')

#tag = T1.iloc[14]

ids=sorted(T1['id'].unique())

#for id in ids[:]:
#id = 2957  # FA
#id = 508
#id = 2750 #FA
#id = 638
#id = 638
#T2 = T1.query('id=={}'.format(id)).copy()

T2=T1.sort_values(by=['id','frame'])

N = T2.shape[0]
T2.index = range(N)

#T2 = T2.iloc[0:200]
N = T2.shape[0]
T2.index = range(N)

#N = min(N, 25)


#%%

imgs = np.zeros((90,90,3)+(N,))
imgs2 = np.zeros((90,90,3)+(N,))

print('Extracting...')
for i,tag in T2.iterrows():
    if i>=N: break
    if (i%10==0): print(i,'/',N)
    tag_img = extract_tag_image(tag, vidcap, True)
    imgs[:,:,:,i] = tag_img/255
    tag_img = extract_tag_image(tag, vidcap, False)
    imgs2[:,:,:,i] = tag_img/255
    
#%%
import h5py

with h5py.File('imgs.h5', 'w') as hf:
    hf.create_dataset("imgs",  data=imgs, compression="gzip")
    hf.create_dataset("imgs2",  data=imgs2, compression="gzip")
    
#%%
import h5py
    
with h5py.File('imgs.h5', 'r') as hf:
    imgs = hf['imgs'][:]
    
#%%
def extract_color_features(imgs, df):
    means = imgs.mean(axis=(0,1)).transpose()
    df['rgb_mean'] = means.tolist()

extract_color_features(imgs, T2)

R=T2[['rgb_mean']].applymap(lambda rgb: rgb[0]).as_matrix()
G=T2[['rgb_mean']].applymap(lambda rgb: rgb[1]).as_matrix()
B=T2[['rgb_mean']].applymap(lambda rgb: rgb[2]).as_matrix()
C=T2['rgb_mean'].as_matrix()
    
#%%

Y=0.299*R+0.587*G+0.1114*B
CB=(B-Y)/Y
CR=(R-Y)/Y

fig=plt.figure()
plt.scatter(CB,CR,c=C,marker='+')
plt.plot(0.05,-0.05,'*')
plt.plot([0.05,0.05+0.05],[-0.05,-0.05+0.05],'-')
plt.axis('equal')

fig=plt.figure()
plt.hist(CB-CR,30)

#%%
print('Saving...')
plot_tag_grid_pdf(imgs, 'tags.pdf', T2)
        
#%%
plot_tag_grid_png(imgs2, imgs, 'tags', T3)

#%%
plot_tag_grid_png(imgs2, imgs, 'tags_P5', T3)
    
#%%
fig,ax=plt.subplots(1,2)
ax[0].imshow(imgs.mean(3))
ax[1].imshow(np.sqrt(imgs.var(3).mean(2)))
    
#%%
def plot_tag_grid(imgs):
    N = imgs.shape[3]
    
    nx=int(np.ceil(np.sqrt(N)))
    ny=int(np.ceil(N/nx))
        
    fig,ax = plt.subplots(ny,nx,squeeze=False,figsize=(nx*2,ny*2))
    
    ax=ax.ravel()
    for i in range(N,nx*ny):
        ax[i].set_axis_off() #remove()
    
    for i in range(N):
        tag_img = imgs[...,i]
        ax[i].imshow(tag_img)
        ax[i].set_xticks([]);     ax[i].set_yticks([])
        #ax[i].set_title(str(T2.iloc[i]['frame'])+' '+str(round(T2.iloc[i]['dm'])))
        ax[i].set_title(str(round(T2.iloc[i]['dm'])))
        
    return fig

#%%
from skimage import io 
from skimage import img_as_ubyte   

def putMultilineText(I, text, pos, font, dy=12):
    (x,y0)=pos
    for i, line in enumerate(text.split('\n')):
        y = y0 + i*dy
        cv2.putText(I,line,(x,y), font, 1,(255,0,0),1,cv2.LINE_AA)
        # # Caution: BGR order

def plot_tag_grid_png(imgs, imgs2, filename, df=None):
    nx=10
    ny=10
    
    old_id=None

    N = imgs.shape[3]
    
    S = imgs.shape[0] # also shape[1]
    
    XX = nx*S
    
    knum = (N + (nx*ny-1)) // (nx*ny)
    for k in range( knum ):
        
        print(k,'/',knum-1)
        
        I = np.ones( (S*ny,S*nx*2,3) , dtype='uint8')
        
        for i in range(k*nx*ny, min(N,(k+1)*nx*ny)):
            j = i-k*nx*ny
            print(i,j)
            
            X = (j % nx)*S
            Y = (j // nx)*S
            
            id=df.iloc[i]['id']
            if (id==old_id):
                mark=False
            else:
                mark=True
                old_id=id
            
            I[Y:Y+S,X:X+S,:] = imgs[...,i]*255
            I[Y:Y+S,XX+X:XX+X+S,:] = imgs2[...,i]*255
            if (mark):
                I[Y:Y+S,X,:] = [255,0,0] 
                
            if (df.iloc[i]['dm'] <= 20):
                I[Y+S-10:Y+S,X:X+10,:] = [255,0,255] 
                
            if 'rgb_mean' in df.columns:
                rgb_mean=df.iloc[i]['rgb_mean']
                y=0.299*rgb_mean[0]+0.587*rgb_mean[1]+0.1114*rgb_mean[2]
                CB=(rgb_mean[2]-y)/y
                CR=(rgb_mean[0]-y)/y
                if (CB-CR > 0.35):
                    I[Y+S-10:Y+S,X+S-10:X+S,:] = [0,0,255] 
            
            if (df is not None):
                text = 'id{}\nf{}\ndm{:.0f}'.format(df.iloc[i]['id'],
                                              df.iloc[i]['frame'],
                                              df.iloc[i]['dm'])
                
                font = cv2.FONT_HERSHEY_PLAIN
                putMultilineText(I,text,(X,Y+12), font)

        #plt.imshow(I)
        if (filename is None):
            plt.imshow(I)
        else:
            io.imsave(filename+'-'+str(k)+'.png',I)

#%%
from matplotlib.backends.backend_pdf import PdfPages

def plot_tag_grid_pdf(imgs, filename, df):
    nx=5
    ny=4
    
    old_id=None
    
    with PdfPages(filename) as pdf:
        N = imgs.shape[3]
        
        #nx=int(np.ceil(np.sqrt(N)))
        #ny=int(np.ceil(N/nx))
        
        for k in range( (N + (nx*ny-1)) // (nx*ny) ):
            
            fig,ax = plt.subplots(ny,nx,squeeze=False,figsize=(nx*2,ny*2))
            
            ax=ax.ravel()
            for i in range(N-k*nx*ny,nx*ny):
                ax[i].set_axis_off() #remove()
            
            for i in range(k*nx*ny, min(N,(k+1)*nx*ny)):
                j = i-k*nx*ny
                print(i,j)
                tag_img = imgs[...,i]
                
                id=df.iloc[i]['id']
                if (id==old_id):
                    mark=''
                else:
                    mark='* '
                    old_id=id
                
                ax[j].imshow(tag_img)
                ax[j].set_xticks([]);     ax[j].set_yticks([])
                ax[j].set_title(mark+'id'+str(df.iloc[i]['id'])
                                + ' f'+str(df.iloc[i]['frame'])
                                +' dm'+str(round(df.iloc[i]['dm'])),
                                {'fontsize': 9})
                #ax[j].set_title(str(round(T2.iloc[i]['dm'])))
            #fig.suptitle('id={}'.format(id))
            pdf.savefig(fig)
            plt.close(fig)


#%%
fig = plt.figure()
plt.imshow(gray)
plot_tags(T2, ax=None, labels=T2['frame'])

#%%

frame = tag.frame

vidcap.set(cv2.CAP_PROP_POS_MSEC,1000.0/fps*(frame-1))
status,orig = vidcap.read();

gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
rgb = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)

T2 = T1.query('frame=={}'.format(frame))
print('frame {}: {} tags'.format(frame,T2.shape[0]))


##%%

plt.figure()
plt.imshow(gray)
plot_tags(T2, ax=None, labels=None)


#%%
import skimage

frame = 48033 # Two tags on 48033
fps=20
    
vidcap.set(cv2.CAP_PROP_POS_MSEC,1000.0/fps*(frame-1))
status,orig = vidcap.read();

#gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
rgb = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
gray1 = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
gray2 = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

gray3 = skimage.color.rgb2gray(rgb)
    
fig, axes = plt.subplots(2,2)
axes[0,0].imshow(gray1,cmap='gray')
axes[0,1].imshow(gray2,cmap='gray')

axes[1,0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
axes[1,1].imshow(gray3, cmap='gray')

#%%

import apriltagdetect as at

det = at.init_detection(config='tag25h5inv')
detections = at.do_detect(det, orig)

obj = at.detectionsToObj(detections)

#%%
plt.figure()
plt.imshow(rgb)
plot_tags(df,labels=df['g'].apply(lambda g: round(g,1)))

#plot_tags(T2.query('frame=={}'.format(frame)), ax=None, labels=None)

#%%

import matplotlib.gridspec as gridspec
#fig = plt.figure()

plt.clf()


gs = gridspec.GridSpec(2, 2, width_ratios=[2,1])
#ax = [plt.subplot(gs_i) for gs_i in gs]
ax = [plt.subplot(gs_i) for gs_i in [gs[:,0],gs[0,1],gs[1,1]]]


ax[0].imshow(gray, cmap='gray')
atd.plot_detections(tags,ax[0], orig, labels=range(len(tags)))




id=21

pixsize = 10
S = 9*pixsize
s = pixsize
pts_src = tags[id].corners
#pts_dst = np.array([[0,0],[S,0],[S,S],[0,S]])
pts_dst = np.array([[s,s],[S-s,s],[S-s,S-s],[s,S-s]])
size = (S,S)


h, status = cv2.findHomography(pts_src, pts_dst)
im2 = cv2.warpPerspective(rgb, h, size)

ax[1].imshow(im2)

xgrid,ygrid = np.meshgrid((np.arange(9)+0.5)*s,(np.arange(9)+0.5)*s)
ax[1].plot(xgrid.ravel(),ygrid.ravel(),'+r')
ax[1].plot(pts_dst[[0,1,2,3,0],0],pts_dst[[0,1,2,3,0],1],'-g')



R = 50
S = 100
C = tags[id].corners.mean(0)
pts_src = C + np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*R
pts_dst = np.array([[0,0],[S,0],[S,S],[0,S]])
size = (S,S)

h, status = cv2.findHomography(pts_src, pts_dst)
im3 = cv2.warpPerspective(rgb, h, size)

ax[2].imshow(im3)
