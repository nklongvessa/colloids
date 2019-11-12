# -*- coding: utf-8 -*-
"""
Spyder Editor

My first script for tracking particles. 

-> Open image as an intensity matrix via pims
-> Processing via trackpy 
-> Get positions matrix from every frames and save
-> Reconstract trajectories and save
"""

## Import relevant libraries
from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3


import matplotlib as mpl
import matplotlib.pyplot as plt

#%matplotlib inline # magic command for showing a figure

# Optionally, tweak styles.
mpl.rc('figure',  figsize=(10, 6))
mpl.rc('image', cmap='gray')




import numpy as np # 
import pandas as pd
from pandas import DataFrame, Series  # for convenience

import pims # Python Image Sequence library
import trackpy as tp
import os



size = 13#47 # estimate particle size in pixel (diameter)
mm = 200#400#200 # minmass
maxdis = 7 # max displacement
mem = 5 # allow particle to disappear for some period
#Nframes = 2000 # no. of frames per one activity
threshold = 1
mpp = 0.171
fps = 5
nlagtime = 60


path = "/data4To/Ong/Tracking result/Aderito+Lea/2019_11_05 Dilute exp"
#impath = "/data12To/exp videos/18_03_22 Dense test/{:02d}_/*.tiff" 
impath = "/data12To/exp videos/Aderito+Lea/2019_11_05 Dilute exp/{:02d}/*.tiff" 
savepath_position = os.path.join(path,"Position_Act{}.csv")
savepath_traj_before_drift = os.path.join(path,"Traj_Act{}.csv")
savepath_traj_f = os.path.join(path,"Traj_f_Act{}.csv")
        




select_act = [1,2,3,4,9,10]
for ind, act in enumerate(select_act): # loop for various activities
    
    
    print("Activity", act)
    
      
    frames = pims.ImageSequence(impath.format(1),as_grey= False,process_func = None)
   
    
    ## Locate features

# =============================================================================
#     f = tp.locate(frames[400], size, minmass=mm ,smoothing_size = size, threshold =  2,percentile = 10, characterize = True) # 'look for bright' position, number indicates size of particle (in pixel)
# 
#     
#     # plot to see the first trial
#     plt.figure('tp')  # make a new figure
#     tp.annotate(f, frames[400]); 
#     
#                 
#     ## Mass cutoff (to eliminate fake particles)
#     fig, ax = plt.subplots()
#     ax.hist(f['mass'], bins=20)
#     ## Optionally, label the axes.
#     ax.set(xlabel='mass', ylabel='count');
#        
#     ## Subpixel accuracy
#     plt.figure()
#     tp.subpx_bias(f);
# =============================================================================


    # Collect all detected positions (from every selected frame)
    f = tp.batch(frames[:], size, minmass=mm ,smoothing_size = size, threshold =  2,percentile = 10, characterize = True);
             

    # Save position data
    f.to_csv(savepath_position.format(act), index = False) # don't want an  extra column of index
    #f = pd.read_csv(savepath_position.format(act))
    
    ## Reconstruct trajectories
    traj = tp.link_df(f, maxdis, memory=mem) 
    
# =============================================================================
#     plt.figure()
#     tp.plot_traj(traj);
# =============================================================================
    
    # Save trajectories
    traj.to_csv(savepath_traj_before_drift.format(act), index = False)
# ======================================================================
    
# =============================================================================
#     drift = tp.compute_drift(traj, smoothing= 50)
#     traj_f = tp.subtract_drift(traj, drift)
#     
#     traj_f.to_csv(savepath_traj_f.format(act), index = False)
#     
# =============================================================================
    
    f = [] # clear f
    traj = []
    traj_f = []


