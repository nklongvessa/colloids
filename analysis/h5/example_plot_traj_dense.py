#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 18:24:49 2019

@author: nklongvessa
"""

import os
import sys
sys.path.insert(0, "/data4To/Ong/test python/examples/sand box Trackpy")
from plots_streaming import plot_traj
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec


plt.figure("traj nearfiled x_GB = 50", figsize = (7,3.5))
plt.clf()
plt.rc('font',size = 10)

gs = gridspec.GridSpec(1,1, left = 0.1, right = 0.95)


path = '/data4To/Ong/Tracking result/19_04_17 Dropping Glass Beads/tracking/profile/'
path_traj = os.path.join(path,'Traj_f_Act{}.h5')

fps = 5
mpp = 0.273

select_act = [3]#,5,4]#,2,3,4,5]
select_lg = [40,30,69,80,55] # Delta f


ymin = [118,165]
ymax = [558,605]
xmin = [109,243]
xmax = [577,711]
TeffT0 = [1.0,1.4,5.0]

# time
tstart = [235]
tstop = [311]
t0_0 = [103]
for indt, t in enumerate([1]): # loop t0
       
    print('t{}'.format(t))
    
    for ind, act in enumerate(select_act):
        #plt.figure('Teff/T0 = {} x_GB = 60'.format(TeffT0[ind]))
      
        ax1 = plt.subplot(gs[ind])
        
        
        
        t0 = t0_0[ind]
        t1 = tstart[ind]
        t2 = tstop[ind]
        
        range_frames = (t1,t2,1)
        ROI = None#(xmin[ind], xmax[ind], ymin[ind], ymax[ind])

        
        plot_traj(path_traj.format(act),range_frames, ROI = ROI, mpp = mpp, InvertAxis = True)   
        plt.axis('equal')

        #plt.text(8,11,'$t_0 = {}$ s'.format((t1-t0)/fps), backgroundcolor = 'lightgrey')
      
        if ind == 0:
            plt.title(r'Passive ($T_\mathrm{eff}/T_0 = 1.0$)')
            #plt.title('Passive')
        else:
            plt.title('Active ($T_\mathrm{eff}/T_0 = 6.5$), $\Delta t = 15.2$ s')
            #plt.title('Active')
            
# =============================================================================
#         # add scale bar
# 
#         if ind == 0:
#             trans = ax1.get_xaxis_transform()
#             plt.text(12,95,'10 $\mu$m', rotation= 90 )
#             plt.text(-4.3,110,'_', rotation= 90, size = 60,color = 'dimgrey' )
# =============================================================================
            
# =============================================================================
#         # add gsinetheta    
#         if ind == 0:
#             plt.text(10,45,r'$g\sin\theta$', rotation= 90 )
#             plt.text(12,52,r'$\leftarrow$', rotation= 90, size = 30,color = 'dimgrey' )
#             
# =============================================================================

        plt.xlabel('x [$\mu$m]')
        plt.ylabel('y [$\mu$m]')     
        
        plt.gca().invert_yaxis()
        #plt.xticks([])
        #plt.yticks([])
        #plt.xlim(21,100)
        #plt.ylim(50,110)
        #plt.gca().invert_yaxis()
plt.title('Active')
plt.tight_layout()

#plt.savefig('plot_traj_GB_exp_nearfield_xGB_50.pdf',format = 'pdf', dpi = 350)
#