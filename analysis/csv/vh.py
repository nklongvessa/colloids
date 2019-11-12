#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:10:58 2017
Use the code from trackpy
Calc Van Hove function
@author: nklongvessa
"""
import numpy as np # 
import pandas as pd
from matplotlib import pylab as plt

cmap = plt.cm.cool

mpp = 0.273
maxact = 9
lagtime = 200
bins = 50
select_lt = [5]#[5,20,40,80,180]#,40,200]
strlabel = ["0u", "1u", "2u", "4u", "8u", "16u", "20u", "32u", "40u","64u"]

savepath_traj_f = "/data4To/Ong/Tracking result/17_02_14 Gold H2O2/Traj_f_Act{}.cvs" # final 

plt.figure('van Hove function')
plt.clf()


select_act = [1,2,3,4,5,6,7,8,9]
for act in select_act:
    
    traj_f = pd.read_csv(savepath_traj_f.format(act)) 
    
    pos = traj_f.set_index(['frame', 'particle'])['x'].unstack()
    
    
    for lagtime in select_lt:
        print('lagtime =', lagtime)
    
    
        pos = pos.reindex(np.arange(pos.index[0], 1 + pos.index[-1]))
        assert lagtime <= pos.index.values.max(), \
            "There is a no data out to frame %s. " % pos.index.values.max()
        disp = mpp*pos.sub(pos.shift(lagtime))
        # Let np.histogram choose the best bins for all the data together.
        values = disp.values.flatten()
        values = values[np.isfinite(values)]
        global_bins = np.histogram(values, bins=bins)[1]
        # Use those bins to histogram each column by itself.
        
        vh = disp.apply(
            lambda x: pd.Series(np.histogram(x, bins=global_bins, range=(global_bins.min(),global_bins.max()), density=True)[0]))
        
        vh.index = global_bins[:-1]   
        vhem = vh.sum(1)/len(vh.columns)
    
        plt.plot(vhem,color=cmap((act-1)/maxact),label = strlabel[act-1])

    

ylable = 'G(x,$\Delta t = {}$)'
plt.ylabel(ylable.format(lagtime))
plt.xlabel('$\Delta x$ ($\mu m$)')

plt.show()
