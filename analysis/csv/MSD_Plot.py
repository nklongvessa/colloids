#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 10:24:03 2016

Plot ensemble MSD

@author: nklongvessa
"""
from matplotlib import pylab as plt
import numpy as np 
import pandas as pd

import os

cmap = plt.cm.cool # color map

#strlabel = "Activity {}"
strlabel = ["0u","0.1u", "0.5u","1u", "2u", "4u", "8u", "16u", "32u","64u", "128u"]
#path = "/data2/Ong/Tracking result/18_03_23 Dense exp 6/fps 5"
path = "/data4To/Ong/Tracking result/17_06_06 Dense exp 3/dense 5 fps/graph+etc"
#path = "/data2/Ong/Tracking result/17_06_06 Dense exp 3/dense 5 fps/graph+etc"

savepath_MSD = os.path.join(path,"MSD_Act{}.cvs")


markers = ['o','v','*','>','p','s','h','^','8','D','<']

plt.figure()
plt.clf()

maxact = 10

select_act = range(1,maxact+1)#[1,3,6,5,2,7,9,4,8,10]
for ind,act in enumerate(select_act): #range(1,9): 
    
    print('Activity', act)
    
    em = pd.read_csv(savepath_MSD.format(act)) # load MSD files
    #em.rename(columns={"Unnamed: 0":"lagt"},inplace = True)
        
    # plt.plot(em.index, em, 'o-', color=cmap((act-1)/8), mec=(0,0,0,0.5), label = strlabel.format(act)) # detail = False
    plt.plot(em['lagt'], em['msd'], '-', color=cmap((act)/maxact), mec=cmap((act)/maxact),marker=markers[ind], label = strlabel[ind], markersize=5)#.format(act)) # detail = True
    #plt.errorbar(em['lagt'],em['msd'],yerr = em['std'],color=cmap((act)/maxact))
    #plt.plot(em['lagt'],np.abs(em['<x^2>']-em['<y^2>']), color=cmap((act)/maxact))
#    plt.plot(em['lagt']/5, em['msd'], '-', color=cmap((v[act]-2.23545)/0.46284),marker=markers[ind], label = strlabel.format(vel['mean'][act]), markersize=6)#
#==============================================================================
#     # <x^2>,<y^2>
#    plt.plot(em['lagt'], em['<x^2>'], '-', color=cmap((act)/maxact),mec=(0,0,0,0.5), label = '<x^2>') # detail = True
#    plt.plot(em['lagt'], em['<y^2>'], '-', color=cmap((act)/maxact),mec=(0,0,0,0.5), label = '<y^2>') 
#==============================================================================
    
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]')
plt.xlabel('lag time $t$ [s]')
#ax.set(ylim=(1e-2, 10));
t2 = np.logspace(-0.2, 0.7, base=10)
plt.plot((t2),200*(t2)**2, '--k') # balistic

t1 = np.logspace(2, 3, base=10)
plt.plot((t1+4), 0.6*(t1+4), 'k') # diffusive
plt.plot((t1), 0.6*(t1), 'k') # diffusive

    
#t1 = np.logspace(-2, -1, base=10)
#plt.plot((t1+0.02), 0.5*(t1+0.02), 'k') # diffusive
#plt.text(0.2,1.4,"2")
#plt.text(0.05,0.03,"1")
#plt.text(256,1.9,"1")

#==============================================================================
#  # With linear fit
# plt.figure()
# plt.ylabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]')
# plt.xlabel('lag time $t$');
# tp.utils.fit_powerlaw(em)
#==============================================================================

#plt.savefig('/data2/Ong/Tracking result/16_11_28 H2O2/MSD_AllAct.pdf')
plt.show()

# find Deff
#==============================================================================
# 
# import scipy
# maxact = 10
# plt.figure('MSD lin-lin')
# plt.clf()
# xminar = [40, 40, 40, 40, 40, 40, 40, 40 ,40, 45,45]
# xmaxar = [67, 67, 67, 67, 67, 67, 67, 67,67,67,67] 
# deff = pd.DataFrame(np.zeros((maxact,2)),columns = ['Deff', 'std_err'])
# select_act = [1,3,4,5,6,8]
# for act in range(1,11): 
#     
#     print('Activity', act)
#     
#     em = pd.read_csv(savepath_MSD.format(act)) # load MSD files
#     xplot = em['lagt']
#     yplot= em['msd']#-(fit1[0]*em['lagt']+fit1[1])
#     # plt.plot(em.index, em, 'o-', color=cmap((act-1)/8), mec=(0,0,0,0.5), label = strlabel.format(act)) # detail = False
#     plt.plot(xplot,yplot, 'o-', color=cmap((act-1)/maxact), mec=cmap((act)/maxact), label = strlabel[act-1], markersize=3)#.format(act)) # detail = True
#     
#     xmin = xminar[act-1]
#     xmax = xmaxar[act-1]
#     #fit = np.polyfit(xplot[xmin:xmax],yplot[xmin:xmax],1)
#     m, b, r_value, p_value, std_err = scipy.stats.linregress(xplot[xmin:xmax],yplot[xmin:xmax])
#     plt.plot(xplot[xmin:xmax],m*xplot[xmin:xmax] + b,'k')
#     
#     deff.Deff[act-1] = m/4
#     deff.std_err[act-1] = std_err
#     #deff[act -1] = (em.msd.max()/em.lagt.max())/4
# 
# plt.ylabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]')
# plt.xlabel('lag time $t$')
# 
# 
# plt.figure()
# plt.clf()
# 
# conc = [0,0.006,0.01,0.03,0.05,0.1]
# conc= np.array([0,2,4,8,16,32])
# plt.plot(conc/300,deff.Deff[1,3,4,5,6,8],'o')
# plt.errorbar(conc/300,deff.Deff, yerr = deff.std_err,markersize = 0)
# 
#==============================================================================
#==============================================================================
# # Fit the whole MSD curve 
# 
# from scipy.optimize import curve_fit
# 
# 
# lagt = em['lagt'].values
# tauRfit = np.zeros(maxact)
# 
# plt.figure()
# plt.clf()
# for act in range(1,maxact+1):
#     
#     v0 = v0norm[act-1]
#     em = pd.read_csv(savepath_MSD.format(act)) # load MSD files
#     ydata = em['msd'][:300].values
# 
# 
#     def func(lagt, tauR): # Fitting function
#         D0 = 0.2168
#         v0 = v0norm[act-1]
#         return 4*D0*lagt+(1/2)*v0**2*tauR**2*(2*lagt/tauR+np.exp(-2*lagt/tauR)-1) 
#         
# 
# 
# 
#     popt, pcov = curve_fit(func, lagt[:300], ydata)
#     tauRfit[act-1] = popt
#     
#     
#     yplot = func(lagt[:300],popt)  
#     plt.plot(lagt[:300],yplot,color=cmap((act)/maxact))
#     
#   
#   
# 
# plt.figure()
# plt.clf()
# plt.plot(conc[0:9],tauRfit)
# plt.plot(conc[0:9],((deff-deff[0])*4/(np.array(v0norm)**2)))
# plt.plot(conc[0:9],np.array(v0norm))
# plt.plot(conc[0:9],deff)
# 
#   
#==============================================================================
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  