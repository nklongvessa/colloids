#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:08:19 2016
Calculate MSD 
@author: nklongvessa
"""

from matplotlib import pylab as plt
import trackpy as tp
import pandas as pd


cmap = plt.cm.autumn

savepath_traj_f = "/data2/Ong/Tracking result/18_02_07 LatexAuNP Janus/Active 6u/Traj_f_Act{}.csv" # final trajectory
savepath_MSD = "/data2/Ong/Tracking result/18_02_07 LatexAuNP Janus/Active 6u/MSD_Act{}.csv"

strlabel = "Activity {}"
mpp = 0.274 # microns per pixel (0.273 with 1.6x, 0.171 with 1x)
fps = 25 # frame per second 
maxact = 11

##fig, ax = plt.subplots() # for ploting msd curve of all activites
#plt.figure('MSD for aactivities')
#plt.clf()

for act in range(8, 12): # loop for file name

    print('Activity', act)
    # load trajectory
    traj_f = pd.read_csv(savepath_traj_f.format(act)) 



#==============================================================================
#      ## Mean Squared Displacement of Individal Probes   
#      im = tp.imsd(traj_f, mpp, fps, max_lagtime=100, statistic='msd')  # microns per pixel = 100/285., frames per second = 24
#      
#      fig, ax = plt.subplots()
#      ax.plot(im.index, im[18], 'k-', alpha=0.1)  # black lines, semitransparent
#      ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]', xlabel='lag time $t$')
#      ax.set_xscale('log')
#      ax.set_yscale('log')
#==============================================================================
      
    
    ## Ensemble MSD
    em = tp.emsd(traj_f, mpp, fps,max_lagtime=1600, detail=True) #detail=True to see also <x>,<y>,<x^2>,<y^2>
    
    # save MSD
    em.to_csv(savepath_MSD.format(act),index = False)
