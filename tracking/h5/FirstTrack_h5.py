# -*- coding: utf-8 -*-
"""
Spyder Editor

Streaming particle tracking with HDF5 files (.h5)

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




#import numpy as np # 
#import pandas as pd

import scipy 
import pims # Python Image Sequence library
import trackpy as tp

import os



size = 5#47 # estimate particle size in pixel (diameter)
mm = 0#400#200 # minmass
smoothing_size = 3
separation = 2
threshold = 2
percentile = 2
maxdis = 4#6 # max displacement
mem = 5 # allow particle to disappear for some period
#Nframes = 2000 # no. of frames per one activity
mpp = 0.273
fps = 5




savepath = "/data4To/Ong/Tracking result/19_04_17 Dropping Glass Beads/tracking"
savepath_position = os.path.join(savepath, 'profile_refined/Position_Act{}.h5')
savepath_traj = os.path.join(savepath, 'profile_refined/Traj_f_Act{}.h5')

#impath = "/data12To/exp videos/18_03_22 Dense test/{:02d}_/*.tiff" 
impath = "/data12To/exp videos/Selva/darkfield/active 17_05/{}/*.tiff" 
impath_act = ['Passive 01', 'passive 02/New folder (4)', 'exp 1_20u', 'exp 3_ 30u', 'exp 5_10u']

rotate_angle = [-4.2,-1.53,-2.04,-5.8,0]

# Teff
xmin = [1020,1362,200,810,800]
xmax = [1670,1802,1600,1600,1380]
ymin = [1140,1488,50,1040,123]
ymax = [2000,2038,800,1850,1000]
fmin = [3,0,0,0,1385]
fmax = [2294,1860,2752,2708,2567]



select_act = [1,2,3,4,5]
for ind, act in enumerate(select_act): # loop for various activities
   

    def crop(img):
        """
        Crop the image to select the region of interest
        """
        para = [xmin[ind], xmax[ind], ymin[ind], ymax[ind], rotate_angle[ind]]
        img_rotate = scipy.ndimage.rotate(img,angle = para[4]) 
        x_min = para[0]
        x_max = para[1]
        y_min = para[2]
        y_max = para[3]
        return img_rotate[y_min:y_max,x_min:x_max]         
    
    
    print("Activity", act)
    
    # Balmer camera
    #filename = [f for f in os.listdir(filepath) if f.startswith(actnum.format(act)) and f.endswith(".tif")] # get filename in one activity
    #fulldir = [os.path.join(filepath,filename[i]) for i in range(0,Nframes+1)] # filepath + filename                
    #frames = pims.ImageSequence(fulldir) # read file name of all activities

    #frames = pims.ImageSequence(filepath.format(act),as_grey= False,process_func = None)
    
    frames = pims.ImageSequence(impath.format(impath_act[ind]), process_func = crop)

# =============================================================================
#     plt.figure()
#     plt.imshow(frames[3])
#     
#     
# =============================================================================
    
# =============================================================================
#     
#     ## Locate features
# 
#     f = tp.locate(frames[1860], size, 
#                   minmass=mm,
#                   smoothing_size = smoothing_size, 
#                   separation = separation, 
#                   threshold =  threshold,
#                   percentile = percentile, 
#                   characterize = True) # 'look for bright' position, number indicates size of particle (in pixel)
#     #f = tp.locate(frames[400], size, minmass=mm ,smoothing_size = 45,separation = 100, threshold =  None,percentile=99, characterize = False, invert = True) # 'look for bright' position, number indicates size of particle (in pixel)
# 
#     #f = f[(f['size'] < 2.1) & (f['ecc'] < 0.35) & (f['mass'] < 1000)] # filter 2
#     
#     # plot to see the first trial
#     plt.figure('tp')  # make a new figure
#     plt.clf()
#     tp.annotate(f, frames[1860]); 
# =============================================================================
                
# =============================================================================
#     # estimate density
#     x_min = xmin_act[ind]
#     x_max = xmax_act[ind]
#     y_min = ymin_act[ind]
#     y_max = ymax_act[ind]
#     area =  (x_max-x_min)*(y_max-y_min)*mpp**2
#     phi = len(f1)*np.pi*1.00**2/area
#     phi
# =============================================================================

                
                
# =============================================================================
#     ## Mass cutoff (to eliminate fake particles)
#     fig, ax = plt.subplots()
#     ax.hist(f['mass'], bins=20)
#     ## Optionally, label the axes.
#     ax.set(xlabel='mass', ylabel='count');
#     
#     fig, ax = plt.subplots()
#     ax.hist(f['size'], bins=20)
#     ## Optionally, label the axes.
#     ax.set(xlabel='size', ylabel='count');
#  
#     fig, ax = plt.subplots()
#     ax.hist(f['ecc'], bins=20)
#     ## Optionally, label the axes.
#     ax.set(xlabel='ecc', ylabel='count');
#        
#     ## Subpixel accuracy
#     plt.figure()
#     tp.subpx_bias(f);
# =============================================================================
    
    
    # Collect all detected positions GB
    with tp.PandasHDFStore(savepath_position.format(act)) as s:
        tp.batch(frames[fmin[ind]:fmax[ind]], size, 
                 minmass=mm ,
                 smoothing_size = smoothing_size, 
                 separation = separation, 
                 threshold =  threshold,
                 percentile = percentile, 
                 characterize = True,
                 output=s)
 
            


# =============================================================================
#     with tp.PandasHDFStoreSingleNode(savepath_positionh5.format(act)) as s:
#         with tp.PandasHDFStoreSingleNode(savepath_temp.format(act)) as temp:
#             Nframe = s.max_frame
#             for f in range(0,Nframe+1):
#                 frame = s.get(f)             
#                 frame_filted = frame[((frame['size'] < 3) & (frame['mass'] <3000) & (frame['ecc'] < 0.3))] # filter 2
#                 frame_filted.drop(frame_filted.columns[[2,3,4,5,6,7]], axis = 1, inplace = True) # keep only x,y,frame,particle
#                 temp.put(frame_filted)
# =============================================================================
  
    with tp.PandasHDFStore(savepath_position.format(act)) as pos:## Reconstruct trajectories
        with tp.PandasHDFStore(savepath_traj.format(act)) as traj:
            for linked in tp.link_df_iter(pos, maxdis, memory=mem):
                traj.put(linked)

   
    

