#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 15:14:09 2017

Density profile for sedimentation exp

@author: nklongvessa
"""

import os
import pandas as pd
from matplotlib import pylab as plt
import numpy as np
import trackpy as tp



#filepath = '/data2/Ong/Tracking result/17_06_06 Dense exp 3/dense 5 fps/densityAct{}.xls'
#filepath = '/data4To/Ong/Tracking result/17_06_06 Dense exp 3/dense 5 fps/Traj_f_Act{}.h5'

path = '/data4To/Ong/Tracking result/19_04_17 Dropping Glass Beads'
path_traj = os.path.join(path,'tracking/profile/Traj_f_Act{}.h5')
savepath = os.path.join(path,'profile/profile/profile_Act{}_bins20.csv')
                                                                                   

markers = ['o','v','*','>','p','s','h','^','8','D','<']

Reff = 1.05            
mpp = 0.273                         
                                   
select_act = [1,2,3,4,5]     
               
                                                    
for ind, act in enumerate(select_act):
    
    result = pd.DataFrame()
    #d = pd.read_csv(filepath.format(act),sep='\t')
    #plt.plot(d['X']+shift[ind], d['Y'], '-', color=cmap((act)/maxact), mec=cmap((act)/maxact), label = strlabel[ind])
    print('Act', act)
    with tp.PandasHDFStore(path_traj.format(act)) as s:
        frames = s.frames
        x = []
        for f in frames:
            print('Frames_{}'.format(f))
            oneframe = s.get(f)
            x = x + list(oneframe['x'].values)                        
        
    x_hist = np.histogram(x, bins = 20)
    area =  (x_hist[1][1]-x_hist[1][0])*(oneframe.y.max()-oneframe.y.min())*mpp**2
    Adensity = (x_hist[0]/len(frames))*np.pi*Reff**2/area  
        
    result['x_px'] = x_hist[1][:-1]
    result['Adensity'] = Adensity
    
    result.to_csv(savepath.format(act),index = False)    
    
# =============================================================================
# plt.figure()
# xplot = result['x_px']
# yplot = result['Adensity']
# plt.plot(xplot, yplot)
# =============================================================================
        

# =============================================================================
#         
# plt.figure()
# plt.clf()
# cmap = plt.cm.cool # colo
# maxact = 11
# strlabel = range(0,11)
# for ind, act in enumerate(range(1,11)):
#     #plt.plot((Adensity.index+shift[ind]/8)*616*0.171/60, Adensity['Act{}'.format(act)], color=cmap((act)/maxact), mec=cmap((act)/maxact), label = strlabel[ind],marker=markers[ind],linestyle='None',markersize = 5)#.format(act)) # detai
# 
#     #plt.plot((Adensity.index+shift[ind]/8)*616*0.171/60, Adensity['Act{}'.format(act)], '-', color=cmap((act)/maxact), mec=cmap((act)/maxact), label = strlabel[ind])#.format(act)) # detai
#     plt.plot((Adensity.index+shift_start[ind])*616*0.171/60 - 87, Adensity['Act{}'.format(act)], '+', color=cmap((act)/maxact), mec=cmap((act)/maxact), label = strlabel[ind])#.format(act)) # detai
#     #plt.plot(np.linspace(-30,-5), np.exp(fit.m[ind]*np.linspace(-30,-5)+fit.c[ind]),color=cmap((act)/maxact))
# 
#     
#     
# # fit the density profile
# 
# act = 9
# Adensity_act = Adensity['Act{}'.format(act)]
# #Adensity_act[:32] = 0
# Adensity_act = np.log(Adensity_act)
# 
# indmin = [41,35,38,36,41,35,24,19,19,33,37]
# indmax = [45,41,42,41,45,42,32,35,32,48,58]
# 
# plt.figure()
# plt.clf()
# 
# plt.plot(Adensity_act.index, Adensity_act, '+', color=cmap((act)/maxact), mec=cmap((act)/maxact), label = strlabel[ind])#.format(act)) # detai
# 
# xplt = Adensity.index*616*0.171/60
# 
# m, b, r_value, p_value, std_err = scipy.stats.linregress(xplt[indmin[act-1]:indmax[act-1]],Adensity_act[indmin[act-1]:indmax[act-1]])
# 
# 
# plt.figure()
# plt.clf()
# 
# plt.plot(Adensity_act.index*616*0.171/60, Adensity_act, '+', color=cmap((act)/maxact), mec=cmap((act)/maxact), label = strlabel[ind])#.format(act)) # detai
# plt.plot(xplt[indmin[act-1]:indmax[act-1]], xplt[indmin[act-1]:indmax[act-1]]*m + b)
# 
# fit.loc[act-1] = [m,b,std_err]
# 
# 
# #==============================================================================
# # for act in range(1,11+1):
# #     Adensity_act = Adensity['Act{}'.format(act)]
# #     #Adensity_act[:32] = 0
# #     Adensity_act = np.log(Adensity_act)
# #     m, b, r_value, p_value, std_err = scipy.stats.linregress(xplt[indmin[act-1]:indmax[act-1]],Adensity_act[indmin[act-1]:indmax[act-1]])
# #     fit.loc[act-1] = [m,b,std_err]
# # 
# #==============================================================================
# 
# 
# # plot fit
# #Adensity = pd.read_csv('/data2/Ong/Tracking result/17_06_06 Dense exp 3/gas 20 fps/graph+etc/Adensity_profile.csv')
# markers = ['o','v','*','>','p','s','h','^','8','D','<']
# 
# xplt = Adensity.index*616*0.171/60
# 
# 
# plt.figure()
# plt.clf()
# for act in range(1,12):
#     m = fit.m[act-1]
#     b = fit.b[act-1]
#     plt.plot(xplt[indmin[act-1]:indmax[act-1]]+(shift_start[act-1])*616*0.171/60 -84.6, np.log(Adensity['Act{}'.format(act)][indmin[act-1]:indmax[act-1]]), marker = markers[act-1],markersize = 5, ls = 'None', color=cmap((act)/maxact), mec=cmap((act)/maxact), label = 'Act{}'.format(act))
#     plt.plot(xplt[indmin[act-1]:indmax[act-1]]+(shift_start[act-1])*616*0.171/60 - 84.6, xplt[indmin[act-1]:indmax[act-1]]*m + b,color=cmap((act)/maxact))
#     #plt.plot(xplt[indmin[act-1]:indmax[act-1]]+(shift[act-1]/8)*616*0.171/60, np.log(Adensity['Act{}'.format(act)][indmin[act-1]:indmax[act-1]]), marker = markers[act-1],markersize = 5, ls = 'None', color=cmap((act)/maxact), mec=cmap((act)/maxact), label = 'Act{}'.format(act))
#     #plt.plot(xplt[indmin[act-1]:indmax[act-1]]+(shift[act-1]/8)*616*0.171/60, xplt[indmin[act-1]:indmax[act-1]]*m + b,color=cmap((act)/maxact))
#     
#     
#     
# =============================================================================
    
    
    
    
    
    
    
    
    