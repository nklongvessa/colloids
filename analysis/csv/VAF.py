#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 16:48:56 2017

Velocity Auto Correlation Function

Use the 'smoothed' trajectory to calc VAF 

@author: nklongvessa
"""

import numpy as np
import pandas as pd
from matplotlib import pylab as plt
cmap = plt.cm.cool # col

savepath_traj_smooth = "/data4To/Ong/Tracking result/17_02_14 Gold H2O2/Traj_freesmooth08_Act{}.cvs" # final 

strlabel = ["0u", "1u", "2u", "4u", "8u", "16u", "20u", "32u", "40u","64u"]

dr = ['dx', 'dy']
mpp = 0.273
fps = 20
max_lagtime_want = 100
maxact = 9
vaf = np.zeros((max_lagtime_want,maxact)) # final result (row - lagtime, col - Activity) 

for act in range(1,maxact +1): #
    print('Activity', act)

    traj_smooth = pd.read_csv(savepath_traj_smooth.format(act))  # download the smoothed trajectory
    meanvdot = np.zeros((max_lagtime_want,traj_smooth['particle'].nunique())) # initialize array to keep mean dot product of velocities
    meanvdot[:] = np.NaN # gives every element to NaN (to use np.nanmean after)
    
    pi= 0
    for pid, ptraj in traj_smooth.reset_index(drop=True).groupby('particle'): # loop particle
        
        vel = ptraj.set_index('frame')[dr]*fps # micron/sec
        vel = vel.reindex(np.arange(vel.index[0], 1 + vel.index[-1])) # put NaN where index is missing
        
        #u = pd.DataFrame(0, index=np.arange(len(vel)), columns=['dx','dy']) # initialize traj_smooth
        u = np.zeros((len(vel),2))
        u[:,0] = (vel['dx']/np.sqrt((vel.values**2).sum(-1))).values # to normalize velocity
        u[:,1] = (vel['dy']/np.sqrt((vel.values**2).sum(-1))).values 
                                    
        max_lagtime = min(max_lagtime_want, len(u) - 1) # check if the max_lagtime is more than the time that particle exists
        lagtimes = np.arange(1, max_lagtime + 1)
        
        for lt in lagtimes: # loop delta t
            #vdot = [np.dot(vel[:-lt].loc[i],vel[lt:].loc[i+lt]) for i in range(vel.index.min(),vel.index.max()+1-lt)] # normalize doi ni!!!
            vdot = (u[:-lt]*u[lt:]).sum(-1) # dot product between u(t).u(t+dt)
            
            if np.nansum(vdot) == 0: # to prevent mean of empty array
                meanvdot[lt-1,pi] = np.NaN
            else:
                meanvdot[lt-1,pi] =  np.nanmean(vdot)
        
        pi +=1
    
    vaf[:,act-1] = np.nanmean(meanvdot,1)
    logvaf = np.log(vaf)


#np.save('/data2/Ong/Tracking result/17_02_14 Gold H2O2/vaf', vaf)


# for plotting
plt.figure()
plt.clf()
tauR = np.zeros(maxact)
v0fit = np.zeros(maxact) # useless if we use normalized velocity
xmin = 10
xmax = 40
x = np.arange(1, max_lagtime_want + 1)/fps
for act in range(4,10): #maxact+1): #

    plt.plot(np.arange(1, max_lagtime_want + 1)/fps,logvaf[:,act-1],'.--',color=cmap((act-1)/maxact),label = strlabel[act-1])
    
    fit = np.polyfit(x[xmin:xmax],logvaf[xmin:xmax,act-1],1)
    plt.plot(x[xmin:xmax],fit[0]*x[xmin:xmax]+fit[1],color=cmap((act)/maxact))
    
    v0fit[act-1] = np.exp(fit[1]/2)
    tauR[act-1] = -2/fit[0]

plt.ylabel('log(VAF)')
plt.xlabel('$\Delta t$ (s)')
plt.title('Velocity Auto-correlation function')

plt.show()

#==============================================================================
# # Plot 
# plt.figure()
# plt.clf()
# plt.plot([np.linalg.norm(v0[i,:]) for i in range(2,9)],tauR[2:9],'x--')
# plt.plot(0,2.868,'xr')
# plt.ylabel('$\tau_R$ (s)')
# plt.xlabel('$v_0$ ($\mu /s$)')
#==============================================================================


#==============================================================================
# plt.figure()
# plt.clf()
# plt.plot([np.linalg.norm(v0[i,:]) for i in range(2,9)],v0fit[2:9],'o--')
# plt.plot(v0fit[2:9],v0fit[2:9])
# plt.ylabel('$v_0$ fit from VAF')
# plt.xlabel('$v_0$')
#==============================================================================


#==============================================================================
# plt.figure('Compare $v_0$, $\tau_R$ and $D_{eff}$')
# plt.clf()
# plt.plot(deff[2:9],0.2+[np.linalg.norm(v0_smooth08[i,:])**2 for i in range(2,9)]*tauR[2:9]/4,'o--')
# plt.plot(deff[2:9],deff[2:9])
# plt.ylabel('$v_0^2\tau_R/4$')
# plt.xlabel('$D_{eff}$')
#==============================================================================
