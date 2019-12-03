#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 10:19:25 2017

@author: nklongvessa
"""
import numpy as np
import pandas as pd
from trackpy import PandasHDFStoreSingleNode
import trackpy as tp

                

def filter_stubs(path, savepath, threshold = 30, pardump = False, chunksize = 2**15):
    """Filter out trajectories which are shorter than the threshol value. 

    Parameters
    ----------
    path : string
        path to the HDF5 file which contains DataFrames(['particle'])
    savepath: string
        path to be saved the result file as a HDF5 file 
    threshold : integer, default 30
        minimum number of points (video frames) to survive
    pardump : boolean, defaults False
        get the trajectory sizes of all particles at once. 'True' to improve the speed. 
    chunksize : integer, default is 2**15         

    Returns
    -------
    a subset of DataFrame in path, to be saved in savepath
    """
    
    with PandasHDFStoreSingleNode(path) as traj:
        # get a list of particle index
        parindex = traj.list_traj() 
        print('Find trajectories length')
        # initialize a Dataframe [particle index, no. of apperance (frame)] 
        trajsizes = pd.DataFrame(
            np.zeros(len(parindex)),
            index = parindex)
        
        # find the length of each trajectory
        if pardump is True: # able to get all particle index at once
            allpar = traj.store.select_column(traj.key, "particle")
            p = allpar.value_counts()
            trajsizes.loc[p.index, trajsizes.columns] = p
            p = []
            allpar = []
      
        else:
            for chunk in traj.store.select_column(traj.key,"particle", chunksize = chunksize): 
                trajsizes.loc[chunk] += 1 # bin it
        
        # creat a new file to store the result after stubs           
        with PandasHDFStoreSingleNode(savepath) as temp: 
            for f, frame in enumerate(traj): # loop frame
                print('Frame:', f)
                # keep long enough trajectories
                frame = frame[(trajsizes.loc[frame.particle.astype(int)] >= threshold).values]
                #store in temp.h5 file
                temp.put(frame)  
        
            print('Before:', len(parindex))
            print('After:',  len(temp.list_traj()))





def filter_index(path, savepath, pindices):
    """Filter out particle by a set of indices. For HDF5 files.

    Parameters
    ----------
    path : string
        path to the HDF5 file which contains DataFrames(['particle'])
    savepath: string
        path to be saved the result file as a HDF5 file 
    pindices : list 
        list of particle index, to be removed from the DataFrame

    Returns
    -------
    a subset of tracks. Dataframe([x, y, frame, particle]), to be saved in savepath
    """
    
    
    with PandasHDFStoreSingleNode(path) as traj:
        with PandasHDFStoreSingleNode(savepath) as result:
            for f, frame in enumerate(traj): 
                print('Frame:', f)
                frame.set_index('particle', drop=False, inplace = True)
                #to be removed: intersection between particle in frame and pindices
                remove = np.intersect1d(frame.particle.values, pindices)
                frame.drop(remove, inplace = True)
                frame.drop(frame.columns[[2,3,4]], axis = 1, inplace = True) # drop [mass, size, ecc]
                result.put(frame)
                
            print('Before:', len(traj.list_traj()))
            print('After:',  len(result.list_traj()))
    
    

def par_char(path, sample = 20):
    """Get particle mass, size and ecc as a time average value.
    *Note that this is not the average of every frame, it is the average among few frames
    which are selected linearly from the minimum to maximum frames.

    Parameters
    ----------
    path : string
        path to the HDF5 file which contains DataFrames(['mass','size', 'ecc', 'particle'])
    sample : integer, Default 20
        a number of frame to be averaged 
        
    Returns
    -------
    DataFrame([index = particle, mass, size, ecc])
    """

    with PandasHDFStoreSingleNode(path) as traj:
        frame_sam = np.linspace(0,traj.max_frame, sample).astype(int)
        frames = traj.store.select(traj.key, "frame in frame_sam", columns= ['mass', 'size', 'ecc','particle'])
        result = frames.groupby('particle').mean()
      
    return result






def emsd(path, mpp, fps, nlagtime, max_lagtime, framejump = 10, pos_columns=None):
    """Compute the mean displacement and mean squared displacement of one
    trajectory over a range of time intervals for the streaming function.

    Parameters
    ----------
    path : string 
        path to the HDF5 file which contains DataFrames(['particle'])
    mpp : microns per pixel
    fps : frames per second
    nlagtime : number of lagtime to which MSD is computed 
    max_lagtime : maximum intervals of frames out to which MSD is computed
    framejump : integer indicates the jump in t0 loop (to increase the speed) 
        Default : 10

    Returns
    -------
    DataFrame([<x^2>, <y^2>, msd, std, lagt])

    Notes
    -----
    Input units are pixels and frames. Output units are microns and seconds.
    """
    
    if pos_columns is None:
        pos_columns = ['x', 'y']
    result_columns = ['<{}^2>'.format(p) for p in pos_columns] + \
                      ['msd','std','lagt'] 
                      
    # define the lagtime to which MSD is computed. From 1 to fps, lagtime increases linearly with the step 1. 
    # Above fps, lagtime increases in a log scale until max_lagtime.
    lagtime = np.unique(np.append(np.arange(1,fps),(np.logspace(0,np.log10(max_lagtime/fps),nlagtime-fps)*fps).astype(int)))
    
    
    with PandasHDFStoreSingleNode(path) as traj:
        # get number of frames
        Nframe = traj.max_frame 
        # initialize the result Dataframe
        result = pd.DataFrame(index = lagtime, columns = result_columns) 
        
        # loop delta t
        for lg in lagtime: 
            print('lagtime', lg)
            # initialize t0
            lframe = range(0, Nframe + 1 - lg, framejump) 
            # initialize DataFrame for each t0
            msds = pd.DataFrame(
                index = range(len(lframe)),
                columns = result_columns) 
            
            for k,f in enumerate(lframe): # loop t0
                
                frameA = traj.get(f)
                frameB = traj.get(f+lg)
                # compute different position between 2 frames for each particle
                diff = frameB.set_index('particle')[pos_columns] - frameA.set_index('particle')[pos_columns]
                # <x^2>
                msds[result_columns[0]][k] = np.nanmean((diff.x.values*mpp)**2)
                # <y^2> 
                msds[result_columns[1]][k] = np.nanmean((diff.y.values*mpp)**2)
            # <r^2> = <x^2> + <y^2>
            msds.msd = msds[result_columns[0]] + msds[result_columns[1]] 
            # average over t0
            result[result.index == lg] = [msds.mean()]
            # get the std over each t0
            result.loc[result.index == lg,result.columns[3]] = msds.msd.std() 
            
        result['lagt'] = lagtime/fps
          
        return result
    
    
def compute_drift_SingleNode(path, smoothing=0, pos_columns=None):
    """Return the ensemble drift, xy(t).

    Parameters
    ----------
    path : string p
        path to the HDF5 file which contains DataFrames(['x','y','particle'])
    smoothing : integer
        Smooth the drift using a forward-looking rolling mean over
        this many frames.

    Returns
    -------
    drift : DataFrame([x, y], index=frame)
    """
    
    if pos_columns is None:
        pos_columns = ['x', 'y']
       
    # Drift calculation 
    print('Drift calc')
    with PandasHDFStoreSingleNode(path) as traj:
        Nframe = traj.max_frame
        # initialize drift DataFrame     
        dx = pd.DataFrame(data = np.zeros((Nframe+1,2)),columns = ['x','y'])    
        
        for f, frameB in enumerate(traj): # loop frame
            print('Frame:', f)
            if f>0:
                delta = frameB.set_index('particle')[pos_columns] - frameA.set_index('particle')[pos_columns]
                dx.iloc[f].x = np.nanmean(delta.x.values)
                dx.iloc[f].y = np.nanmean(delta.y.values) # compute drift
            #remember the current frame
            frameA = frameB 
        
        if smoothing > 0:
            dx = pd.rolling_mean(dx, smoothing, min_periods=0)
        x = np.cumsum(dx)
    return x


def compute_drift(path, smoothing=0, pos_columns=None):
    """Return the ensemble drift, xy(t).

    Parameters
    ----------
    path : string p
        path to the HDF5 file which contains DataFrames(['x','y','particle'])
    smoothing : integer
        Smooth the drift using a forward-looking rolling mean over
        this many frames.

    Returns
    -------
    drift : DataFrame([x, y], index=frame)
    """
    
    if pos_columns is None:
        pos_columns = ['x', 'y']
       
    # Drift calculation 
    print('Drift calc')
    with tp.PandasHDFStore(path) as traj:
        Nframe = traj.max_frame
        # initialize drift DataFrame     
        dx = pd.DataFrame(data = np.zeros((Nframe+1,2)),columns = ['x','y'])    
        
        for f, frameB in enumerate(traj): # loop frame
            print('Frame:', f)
            if f>0:
                delta = frameB.set_index('particle')[pos_columns] - frameA.set_index('particle')[pos_columns]
                dx.iloc[f].x = np.nanmean(delta.x.values)
                dx.iloc[f].y = np.nanmean(delta.y.values) # compute drift
            #remember the current frame
            frameA = frameB 
        
        if smoothing > 0:
            dx = pd.rolling_mean(dx, smoothing, min_periods=0)
        x = np.cumsum(dx)
    return x


def subtract_drift(path, savepath, drift=None):
    """Return a copy of particle trajectories with the overall drift subtracted
    out.

    Parameters
    ----------
    path : string 
        path to the HDF5 file which contains DataFrames(['x','y'])
    savepath : string
        path to be saved the result file as a HDF5 file 
    drift : optional 
        DataFrame([x, y], index=frame) 
        If no drift is passed, drift is computed from traj.

    Returns
    -------
    Dataframe, to be saved in savepath
    """
    if drift is None:
        drift = compute_drift(path)

    with PandasHDFStoreSingleNode(path) as traj_old: 
        with PandasHDFStoreSingleNode(savepath) as traj_new:
            for f, frame in enumerate(traj_old):
                print('Frame:', f)
                frame['x'] = frame['x'].sub(drift['x'][f])
                frame['y'] = frame['y'].sub(drift['y'][f])
                # put in the new file
                traj_new.put(frame)  
                                     
   
