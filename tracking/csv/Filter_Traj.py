#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 15:44:42 2016

Filter trajectories
-> minimum frame of the trajectory
-> filter particles (mass and size)
-> drift substraction

@author: nklongvessa
"""

import trackpy as tp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
#from myfunc import remove_stuck


# ======================================================================
# function for removing non-moving particles
def remove_stuck(traj,size):
    """ Function for removing non-moving particles
    """
    from numpy import sqrt, where
    
    r_min = traj.groupby('particle').first()
    r_max = traj.groupby('particle').last()

    pos_columns = ['x','y']
    dist = r_min[pos_columns] - r_max[pos_columns]
    dist_eu = sqrt(dist['x']**2+dist['y']**2)

    index_remove = dist_eu.index[where(dist_eu < size)]
                              
    traj_new = traj
    for i in range(len(index_remove)):
        traj_new =  traj_new[(traj_new['particle'] != index_remove[i])]
                         
    return traj_new
# =======================================================================


savepath_traj = "/data2/Ong/Tracking result/18_02_07 LatexAuNP Janus/Active 6u/Traj_Act{}.csv"
savepath_traj_f = "/data2/Ong/Tracking result/18_02_07 LatexAuNP Janus/Active 6u/Traj_f_Act{}.csv" # final trajectory

size = 9
maxact = 11

for act in range(8,12): 

    print("Activity", act)
        
# # Load trajectories
    traj = pd.read_csv(savepath_traj.format(act))
 

 
     # Keep particles that last for a given period
    traj1 = tp.filter_stubs(traj, 30) # filter 1
    print('Before:', traj['particle'].nunique()) # to compare
    print('After:', traj1['particle'].nunique())
 
    
 
     # Filter out some particles according to their mass and blah blah blah ...
    plt.figure()
    tp.mass_size(traj1.groupby('particle').mean()); # convenience function -- just plots size vs. mass
    plt.figure()
    tp.mass_ecc(traj1.groupby('particle').mean())
     

    condition = lambda x: ((x['mass'].mean() > 400) & (x['size'].mean() > 2.7) & (x['ecc'].mean() < 0.1))
    traj2 = tp.filter(traj1, condition)  # a wrapper for pandas' filter that works around a bug in v 0.12

    #traj3 = traj1[((traj1['mass'] > 5000) & (traj1['ecc'] < 0.2))] # filter 2 old version
    print('Before:', traj1['particle'].nunique()) # to compare
    print('After:', traj2['particle'].nunique())
    
    
    #    traj2 = traj1
    # Removing stuck particles
    traj3 = remove_stuck(traj1,size)
    print('Before:', traj2['particle'].nunique()) # to compare
    print('After:', traj3['particle'].nunique())
     
     # See the tracking
#==============================================================================
#     plt.figure()
#     tp.annotate(traj3[traj3['frame'] == 0], frames[0]);
#==============================================================================
     
    traj3.drop(traj3.columns[[2,3,4,5,6,7]], axis = 1, inplace = True) # keep only x,y,frame,particle
     
#==============================================================================
#     plt.figure()
#     tp.plot_traj(traj3);
#==============================================================================
     
     
    
    ## Drift 
    # Calculate the drift
    drift = tp.compute_drift(traj,smoothing = 20) # return the cumsum of <dx,dy>
    
#==============================================================================
#     plt.figure() # plot <dx,dy>
#     drift.plot();
#==============================================================================
    
     # Substract the drift
    traj_f = tp.subtract_drift(traj3.copy(), drift) # final trajectories
    no = traj_f.groupby('frame').size() # no. of particle per frame
    print('Activity', act, 'particels', no.mean())

#==============================================================================
#     plt.figure()
#     tp.plot_traj(traj_f); 
#==============================================================================
    
    # save the final trajectories
    traj_f.to_csv(savepath_traj_f.format(act), index = False)
    
    
    
    traj = []
    traj1 = []
    traj2 = []
    traj3 = []
    traj_f = []

