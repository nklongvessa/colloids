#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 09:59:04 2017

@author: nklongvessa
"""
from trackpy import PandasHDFStore


def plot_traj(path, range_frames, ROI = None, mpp = None, InvertAxis = False, label=False,
              cmap=None, ax=None, t_column=None,
              pos_columns=None, chunksize = 2**10):
    """Plot traces of trajectories for each particle.
    Optionally superimpose it on a frame from the video.

    Parameters
    ----------
    traj : DataFrame
        The DataFrame should include time and spatial coordinate columns.
    colorby : {'particle', 'frame'}, optional
    range_frames : Tuple 
        The prefered range of frames in the format of (fmin,fmax,increment)
    ROI : Tuple
        region of interests
    mpp : float, optional
        Microns per pixel. If omitted, the labels will have units of pixels.
    label : boolean, optional
        Set to True to write particle ID numbers next to trajectories.
    superimpose : ndarray, optional
        Background image, default None
    cmap : colormap, optional
        This is only used in colorby='frame' mode. Default = mpl.cm.winter
    ax : matplotlib axes object, optional
        Defaults to current axes
    t_column : string, optional
        DataFrame column name for time coordinate. Default is 'frame'.
    pos_columns : list of strings, optional
        Dataframe column names for spatial coordinates. Default is ['x', 'y'].
    plot_style : dictionary
        Keyword arguments passed through to the `Axes.plot(...)` command

    Returns
    -------
    Axes object
    
    See Also
    --------
    plot_traj3d : the 3D equivalent of `plot_traj`
    """

    import matplotlib.pyplot as plt
    import pandas as pd


    if cmap is None:
        cmap = plt.cm.winter
    if t_column is None:
        t_column = 'frame'
    if pos_columns is None:
        pos_columns = ['x', 'y']
    if mpp is None:
        mpp = 1
    
    frames = range(range_frames[0],range_frames[1],range_frames[2])
    plot_traj = pd.DataFrame()
    with PandasHDFStore(path) as traj_cell:
        
        plot_traj = traj_cell.get(range_frames[0])
        for f in frames: 
            traj_frame = traj_cell.get(f)
            plot_traj = pd.concat([plot_traj,traj_frame])
        
    if ROI is None:
        # get particles index
        parindex = list(set(plot_traj['particle']))  
        if InvertAxis:
            for i in parindex:
                plt.plot(plot_traj[plot_traj['particle'] == i].y*mpp, plot_traj[plot_traj['particle'] == i].x*mpp)
        else:
        
            for i in parindex:
                plt.plot(plot_traj[plot_traj['particle'] == i].x*mpp,plot_traj[plot_traj['particle'] == i].y*mpp)
            
    else: 
        plot_traj = plot_traj[(plot_traj.x > ROI[0]) & (plot_traj.x < ROI[1]) & (plot_traj.y > ROI[2]) & (plot_traj.y < ROI[3])]
        parindex = list(set(plot_traj['particle']))  
        
    # plot trajectories colored by particles
        if InvertAxis:
            for i in parindex:
                plt.plot(plot_traj[plot_traj['particle'] == i].y*mpp - ROI[2]*mpp, plot_traj[plot_traj['particle'] == i].x*mpp - ROI[0]*mpp)
        else:
            
            for i in parindex:
                plt.plot(plot_traj[plot_traj['particle'] == i].x*mpp - ROI[0]*mpp,plot_traj[plot_traj['particle'] == i].y*mpp - ROI[2]*mpp)
                
    plt.show()
       
                
       
    return 


