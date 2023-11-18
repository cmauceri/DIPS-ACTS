import math
import os,sys
import numpy as np
import h5py
import uproot
import awkward as ak
import xarray as xr
import statistics
from sklearn.cluster import DBSCAN
from dbscan1d.core import DBSCAN1D
input_file_path = '/eos/home-p/pibutti/data_acts/acts_ttbar_200PU.root'
trk_output_file_path = 'timesig_clust_output.h5'
#vtx_output_file_path= 'timesig_vtx_output.h5'
chunk_size = 50  # Adjust the chunk size as needed



def processBatch(t,start,stop):
    trk_t=t['track_t'].array(entry_start=start,entry_stop=stop)
    trk_var_t=t['track_var_t'].array(entry_start=start,entry_stop=stop)
    vtxt=t['recovertex_t'].array(entry_start=start,entry_stop=stop)
    vtx_isHS=t['recovertex_isHS'].array(entry_start=start,entry_stop=stop)
    #j_idx=t.arrays('jet_tracks_idx',entry_start=start,entry_stop=stop)    
    #j_label=t.arrays('jet_label',entry_start=start,entry_stop=stop)
    HS_times=[]
    num_trk=0
    nw_trk_t=[]
    nw_trk_var_t=[]

    for ev in range(len(vtxt)):
        evvtxt=vtxt[ev]
        evisHS=vtx_isHS[ev]
        evtrk_t=trk_t[ev]
        evtrk_var_t=trk_var_t[ev]
        nwtrkt,nwtrkvart=doclustering(evtrk_t,evtrk_var_t)
        nw_trk_t.append(nwtrkt)
        nw_trk_var_t.append(nwtrkvart)
        for vtx in range(len(evvtxt)):
            if(evisHS[vtx]==1):
                HS_times.append(evvtxt[vtx])
                break
    trk_t_ak=ak.Array(nw_trk_t)
    trk_var_t_ak=ak.Array(nw_trk_var_t)
    max_tracks = ak.max(ak.num(trk_t_ak))
        
    #for tarr,varr in zip(trk_t_ak,trk_var_t_ak):
        #trk_pad=ak.pad_none(tarr,target=max_tracks,axis=0)
        #var_pad=ak.pad_none(varr,target=max_tracks,axis=0)
        #nw_trk_t.append(trk_pad)
        #nw_trk_var_t.append(var_pad)
    #nw_trk_t_no_none = ak.fill_none(nw_trk_t, -1)
    #nw_trk_var_t_no_none = ak.fill_none(nw_trk_var_t, -1)
    #trk_t_arr=np.array(nw_trk_t_no_none)
    #trk_var_t_arr=np.array(nw_trk_var_t_no_none)
    HS_times_arr=np.array(HS_times)
    #print(HS_times_arr[0])
    significances=calcSig(trk_t_ak,trk_var_t_ak,HS_times_arr)
    return significances
def doclustering(ev_times,ev_vars):
    times = np.asarray(ev_times)
    var=np.asarray(ev_vars)
    dbs = DBSCAN1D(eps=12.5, min_samples=1)
    labels = dbs.fit_predict(times)
    nw_times=times[labels==0]
    nw_vars=var[labels==0]
#    print("time shape:",np.shape(nw_times))

#    nw_times.reshape(-1)
#    nw_vars.reshape(-1)
#    print("var shape:",np.shape(nw_vars))

    return nw_times,nw_vars

def calcSig(times,var,HS):
    sigs=[]
    for ev in range(len(times)):
        HStime=HS[ev]
        evtimes=times[ev]
        evvars=var[ev]
        for t in range(len(evtimes)):
            sig=evtimes[t]-HStime/(np.sqrt(evvars[t]))
            #print("Significance= ",sig)
            sigs.append(sig)
    sigs_arr=np.asarray(sigs)
    return sigs_arr
with uproot.open(input_file_path) as root_file:
    tree = root_file['events']
    total_entries = len(tree['track_t'].array())
    print("events: ",total_entries)
    f=h5py.File(trk_output_file_path, 'a')
    print("before loop")
    #total_chunks = total_entries // chunk_size
    total_chunks=math.floor(total_entries/chunk_size)
    start = 0
    stop = chunk_size
    for i in range(total_chunks):
        print(f"Processing batch {i+1} of {total_chunks}")
        sigs = processBatch(tree, start, stop)
        #print(sigs)
        print("processed batch")
        # Create datasets within the group and write the processed data
        if i==0:
            f.create_dataset("Significance",data=sigs, chunks=True,maxshape=(None,))
            
        else:
            # Append new data to it
            f['Significance'].resize((f['Significance'].shape[0] + sigs.shape[0]), axis=0)
            f['Significance'][-sigs.shape[0]:] = sigs
            
        print("Batch: ", i)
        
        start = stop
        stop = min(stop + chunk_size, total_entries)




















        
