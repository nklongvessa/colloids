#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 14:37:48 2017
Calc 2D Van Hove function from the cross-section of the 2D map
@author: nklongvessa
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:10:58 2017
Use the code from trackpy
@author: nklongvessa
"""
import numpy as np # 
import pandas as pd
from matplotlib import pylab as plt

cmap = plt.cm.cool

mpp = 0.273
fps = 20
maxact = 9
lagtime = 200
select_lt = [5,40,60,100]
bins = 50

sub = '1{}{}' # for the subplot
strlabel = ["0u", "1u", "2u", "4u", "8u", "16u", "20u", "32u", "40u","64u"]

savepath_traj_f = "/data12To/data 2 Tracking result/17_02_14 Gold H2O2/Traj_f_Act{}.cvs" # final 

plt.figure('vh2d')
plt.clf()


select_act = [9]
for act in select_act:
    
    traj_f = pd.read_csv(savepath_traj_f.format(act)) 
    
    
    for lagtime in select_lt:
        print('lagtime =', lagtime)
    
        posx = traj_f.set_index(['frame', 'particle'])['x'].unstack()
        posy = traj_f.set_index(['frame', 'particle'])['y'].unstack()
        
        
        posx = posx.reindex(np.arange(posx.index[0], 1 + posx.index[-1]))
        assert lagtime <= posx.index.values.max(), \
            "There is a no data out to frame %s. " % posx.index.values.max()
        dispx = mpp*posx.sub(posx.shift(lagtime))
        
        posy = posy.reindex(np.arange(posy.index[0], 1 + posx.index[-1]))
        assert lagtime <= posy.index.values.max(), \
            "There is a no data out to frame %s. " % posy.index.values.max()
        dispy = mpp*posy.sub(posy.shift(lagtime))
        
        # find non-active particles and remove them
        find_diff = (abs(dispx)<2) & (abs(dispy)<2)
        ind = (find_diff.sum(0)).index
        if act == select_act[0]: 
            dispx = dispx[ind[np.where(find_diff.sum(0)<50)]]
            dispy = dispy[ind[np.where(find_diff.sum(0)<50)]]   
            keep = ind[np.where(find_diff.sum(0)<50)]
        else:
            dispx = dispx[keep]
            dispy = dispy[keep]       
                          
                          
        # Let np.histogram choose the best bins for all the data together.
        values = dispx.values.flatten()
        values = values[np.isfinite(values)]
        global_bins = np.histogram(values, bins=bins)[1]
        #global_bins = np.arange(values.min(),values.max(),1) # same binsize
        binsize = (global_bins[1:] -global_bins[:-1]).mean()
    
    
        bins = len(global_bins)-1
        vh = np.zeros((bins,bins))
        for i in dispx.columns: # loop for each particle displacement
        
            hist, binsx, binsy = np.histogram2d(dispx[i],dispy[i], bins=(global_bins,global_bins))        
            vh = vh + hist
        
        vh = vh/len(dispx.columns)#/len(dispx.columns)#binsize#sum(vh.sum(0))#
        

        
# =============================================================================
#         plt.subplot(sub.format(len(select_lt),select_lt.index(lagtime)+1)) # ex. subplot(114)
#         plt.imshow(vh, 'hot',interpolation='none', vmin=0,vmax = None, extent=[binsx.min(),binsx.max(),binsy.min(),binsy.max()])
#         plt.colorbar()  
# =============================================================================
        plt.plot(binsx[:-1],vh[25,:])
        

        
        titlelabel = 'G(x,$\Delta t = {}$ s)'
        plt.title(titlelabel .format(lagtime/fps))
        plt.xlabel('$\Delta x$ ($\mu m$)')
        plt.ylabel('$\Delta y$ ($\mu m$)')
        plt.show()

# =============================================================================
# plt.figure('cross vh')
# plt.clf()
# 
# plt.plot(binsx[:-1],vh[25,:])
# 
# =============================================================================
