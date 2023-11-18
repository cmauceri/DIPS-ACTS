import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import load
import h5py
import matplotlib.colors as mcolors

filenames=[]
varnames=['sd0','sz0','st','ptfrac','dr','numPix','numSCT','d0','z0']
fullFile='roc_data/allvar.npy'
colors=['#390099', '#9e0059', '#ff0054', '#ff5400', '#ffbd00']
for var in varnames:
    filenames.append('roc_data/no_'+var+'.npy')
def rocplot(default, neweff,newlabel,newtitle):
    d_leff,d_ceff,d_beff=default
    n_leff,n_ceff,n_beff=neweff
    plt.figure()
    plt.plot()
    # l-rej
    plt.figure()
    plt.plot(n_beff, 1 / n_leff, color=colors[1], label='l-rej (no '+str(newlabel))
    # c-rej
    plt.plot(n_beff, 1 / n_ceff, color=colors[1], linestyle='--', label='c-rej (no '+str(newlabel))
    plt.xlabel('b efficiency')
    plt.ylabel('Background rejection')
   # l-rej
    plt.plot(d_beff, 1 / d_leff, color=colors[0], label='l-rej')
    # c-rej
    plt.plot(d_beff, 1 / d_ceff, color=colors[0], linestyle='--', label='c-rej')
    plt.xlabel('b efficiency')
    plt.ylabel('Background rejection')
    plt.title("ROC without "+str(newlabel)+" and full Inputs")
    plt.legend()
    plt.yscale("log")
    plt.xlim(0.6,1)
    plt.savefig("roc_no_"+str(newtitle+".png"))
    plt.clf()
    return 0
def rocSameCanv(default, effs,labels,varNamesSame):
    d_leff,d_ceff,d_beff=default
    plt.figure()
    plt.plot()
    # l-rej
    plt.plot(d_beff, 1 / d_leff, color=colors[0], label='l-rej')
    # c-rej
    plt.plot(d_beff, 1 / d_ceff, color=colors[0], linestyle='--', label='c-rej')
    plt.xlabel('b efficiency')
    plt.ylabel('Background rejection')
    filestring=''
    print(varNamesSame)

    for j,(eff,newlabel,varname) in enumerate(zip(effs,labels,varNamesSame)):
        n_leff,n_ceff,n_beff=eff
        newtitle='ROC without '
        plt.plot(n_beff, 1 / n_leff, color = colors[j+1],label='l-rej (no '+str(newlabel)+')')
        # c-rej
        plt.plot(n_beff, 1 / n_ceff, color=colors[j+1],linestyle='--', label='c-rej (no '+str(newlabel)+')')
        plt.xlabel('b efficiency')
        plt.ylabel('Background rejection')

        filestring+='_'+varname
    print(filestring)
    newtitle="ROC: full inputs vs removing "+ r'$d_{0}/\sigma_{d0}$'+", "+r'$p_{T}$ fraction'+", and "+r'$\Delta$R'
    plt.title(newtitle)
    plt.legend(borderpad=0.5,fontsize="8",loc="upper right",)
    plt.yscale("log")
    plt.xlim(0.6,1)
    plt.savefig("roc_no"+filestring+".png")
    plt.clf()
    return 0

defEff=load(fullFile)
effs=[]
for filename,var in zip(filenames,varnames):
#    f=h5py.File(filename,"r")
    f=filename
    eff=load(f)
    effs.append(eff)

parameterlist=[r'$d_{0}/\sigma_{d0}$',r'$z_{0}sin{\theta}/\sigma_{z_{0}sin{\theta}}$',r'$t/\sigma_{t}$',r'$p_{T}$ fraction',r'$\Delta$R',r'number of pixel hits',r'number of SCT hits',r'$d_{0}$ [mm]',r'$z_{0}$ [mm]']    
compeffs=effs[0],effs[2],effs[3]
complabels=parameterlist[0],parameterlist[2],parameterlist[3]
compvars=varnames[0],varnames[2],varnames[3]
for eff,name,parameter in zip(effs,varnames,parameterlist):
    rocplot(defEff,eff,parameter,name)    
#rocSameCanv(defEff,compeffs,complabels,compvars)
#f2='roc_data/no_sd0_d0.npy'
#eff2=load(f2)
#name2=r'$d_{0}/\sigma_{d0}$',' and ',r'$d_{\
#0}$ [mm]'
#label2='sd0_d0'
#rocplot(defEff,eff2,label2,name2)
