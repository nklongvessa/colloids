#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 15:59:50 2017

Calculate instantaneous velocity
1. smooth the trajectories: one by one (long)
2. only keep free particles (no other particles in within 'rmin')

@author: nklongvessa
"""
import pandas as pd
from pandas import DataFrame, Series  # for convenience
from matplotlib import pylab as plt
#import trackpy as tp
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
import numpy as np # 
import os

maxact = 10
size = 13
rmin = size*5//2
sigma = 5 # smoothimh parameter
mpp = 0.171 # micrin per pixel
fps = 20
path = "/data4To/Ong/Tracking result/17_02_14 Gold H2O2"
path_traj_f = os.path.join(path,'Traj_f_Act{}.cvs') # final trajectory
savepath_traj_smooth = os.path.join(path,'Traj_smooth_Act{}.cvs') # to save smoothed traj 
savepath_traj_freesmooth = os.path.join(path,'Traj_smooth_Act{}.cvs') # to save free smoothed traj 
pos_columns = ['x', 'y']
dr_columns = ['dx', 'dy']
v0 = np.zeros((maxact,2)) # to collect vo

for act in range(1,maxact+1): 

    traj_f = pd.read_csv(path_traj_f.format(act)) 
    
    traj_smooth = pd.DataFrame(0, index=np.arange(len(traj_f)), columns=['x','y','dx','dy','frame','particle']) # initialize traj_smooth
    traj_smooth['frame'] = traj_f['frame']
    
    # smoothing trajectory --> Gaussian smoothing
    par = list(set(traj_f['particle'])) # get all unique particle index
    
    
    # loop particles
    for p in range(0,len(par)):
        
        print('Smoothing Act{}: {:.2f} %'.format(act,p*100/len(par)))
        #trajp = traj_f[traj_f['particle']== par[p]] # trajectory of one particle
        pos = traj_f.groupby('particle')[pos_columns].get_group(par[p])*mpp # trajectory of each particle (in micron)
    
        # smooth the traj
        xfil = gaussian_filter1d(pos['x'],sigma)
        yfil = gaussian_filter1d(pos['y'],sigma)
    
        # collect smoothed traj in a new dataframe
        traj_smooth.loc[pos.index,'x'] = xfil
        traj_smooth.loc[pos.index,'y'] = yfil
        
#==============================================================================
#         plt.figure()
#         plt.clf()
#         tp.plot_traj(traj_f[traj_f['particle']==par[p]]*mpp)
#         plt.plot(xfil,yfil)
#==============================================================================

        # Calc dx, dy
        dx = [xfil[x] - xfil[x-1] for x in range(1,len(xfil))]
        dx.insert(0,0) # insert '0' at the vx[0]
        dy = [yfil[y] - yfil[y-1] for y in range(1,len(yfil))]
        dy.insert(0,0)
        
        traj_smooth.loc[pos.index,'dx'] = dx
        traj_smooth.loc[pos.index,'dy'] = dy    
        traj_smooth.loc[pos.index,'particle'] = par[p]
        
    traj_smooth.to_csv(savepath_traj_smooth.format(act), index = False)
    
    #traj_smooth = pd.read_csv(savepath_traj_smooth.format(act))

    # Define a free particle
    print('Define free particles')
    traj_free1 = DataFrame()
    notfree = []
    for f in range(0,traj_f['frame'].max()+1):
        #print(f)
        traj_frame = traj_f[traj_f['frame']==f]
        #traj_frame.set_index([[i for i in range(0,len(traj_frame))]],inplace=True)
        tree = cKDTree(traj_frame[['x','y']])
        neigh = tree.query_pairs(rmin)
        neigh = list(neigh)
        neigh_index = [neigh[n][0] for n in range(0,len(neigh))]
        neigh_index = neigh_index + [neigh[n][1] for n in range(0,len(neigh))]
        notfree = notfree + list(traj_frame.index[neigh_index])
                                     
    # only free particle from traj
    traj_freesmooth = traj_smooth.copy()
    traj_freesmooth.drop(traj_freesmooth.index[notfree], inplace=True)
    traj_freesmooth = traj_freesmooth[traj_freesmooth['frame']>0]
    
    traj_freesmooth.to_csv(savepath_traj_freesmooth.format(act), index = False)
    
    # Calc velocity    
    v0[act-1,:] = np.sqrt(((traj_freesmooth[dr_columns]*fps)**2).mean())
    print('v0 = {:f} micron/sec'.format(np.linalg.norm(v0[act-1,:])))
    
v = [np.linalg.norm(v0[i,:]) for i in range(0,10)]


#==============================================================================
# 
# 
# # Calc PDF of v0
# vel = pd.DataFrame(index = range(1,11),columns = ['mean', 'std','N'])
# cmap = plt.cm.cool # color map
# strlabel = "v0 = {:0.3f}"
# markers = ['o','v','*','>','p','s','h','^','8','D']
# plt.figure()
# plt.clf()
# fps = 20
# select_act = [1,3,6,5,2,7,9,4,8,10]
# for ind, act in enumerate(select_act):   
#     traj_freesmooth = pd.read_csv(savepath_traj_freesmooth.format(act))
#     vi2d = traj_freesmooth[dr_columns]*fps
#     vi = np.sqrt(vi2d.dx**2+vi2d.dy**2)
#     h = np.histogram(vi,bins = 30, range = (0,15))
#     vel['mean'][act] = vi[vi.values<20].mean()
#     vel['std'][act] = vi[vi.values<20].std()
#     vel['N'][act] = len(vi)
#     plt.plot(h[1][0:-1],h[0]/len(vi[vi.values<20]), color=cmap((v[act]-2.23545)/0.46284),marker = markers[act-1],markersize = 5,label = strlabel.format(vel['mean'][act]))
# 
#==============================================================================
