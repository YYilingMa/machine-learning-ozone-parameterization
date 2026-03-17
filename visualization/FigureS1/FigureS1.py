###
# plot Figure S1 in the supporting information
# All of them are offline prediction
# compare Temp only, Temp+q, Temp+p
# compare single column, zonal, global domain input
# compare NN and ridge regression offline 
# compare their R2 fields
# NN prediction is produced from FigureS1_Offline_NN.py
# Ridge regression prediction with zonal, global input is from 
# FigureS1_Offline_Zonal_Input.py, FigureS1_Offline_Global_Input.py, respectively

#%% 
import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt
import glob
from matplotlib.ticker import MultipleLocator

Infilepath = "./"
Outfilepath = "./offline_compare/"

if not os.path.exists(Outfilepath):
    os.makedirs(Outfilepath)
    print("The new directory {} is created!".format(Outfilepath))

#%% 
file = xr.open_dataset(Infilepath+"output_single_2000s_40Yrtrain/"+"Offline_R2_score_with_single_column_input_2000s.nc")
R2_original = file["R2_test"].mean(dim=['longitude'],skipna=True).squeeze()
file = xr.open_dataset(Infilepath+"output_single_2000s_40Yrtrain_addQ_1/"+"Offline_R2_score_with_single_column_input_2000s.nc")
R2_addQ = file["R2_test"].mean(dim=['longitude'],skipna=True).squeeze()
file = xr.open_dataset(Infilepath+"output_single_2000s_40Yrtrain_add_P/"+"Offline_R2_score_with_single_column_input_2000s.nc")
R2_addP = file["R2_test"].mean(dim=['longitude'],skipna=True).squeeze()
file = xr.open_dataset(Infilepath+"output_Zonal_Input_ZonalR2/"+"Offline_R2_score_with_zonal_input_2000s.nc")
R2_zonal = file["R2_test"].mean(dim=['longitude'],skipna=True).squeeze()
file = xr.open_dataset(Infilepath+"output_global_R2_coarser/"+"Offline_R2_score_with_global_input_2000s.nc")
R2_global = file["R2_test"].mean(dim=['longitude'],skipna=True).squeeze()
file = xr.open_dataset(Infilepath+"offline_NN_ZonalR2_4/"+"Offline_NN_R2_score_with_single_column_input_2000s.nc")
R2_NN = file["R2_test"].mean(dim=['longitude'],skipna=True).squeeze()
file = xr.open_dataset(Infilepath+"offline_NN_ZonalR2_add_q_4/"+"Offline_NN_R2_score_with_single_column_input_2000s.nc")
R2_NN_addQ = file["R2_test"].mean(dim=['longitude'],skipna=True).squeeze()
file = xr.open_dataset(Infilepath+"offline_NN_ZonalR2_add_p_4/"+"Offline_NN_R2_score_with_single_column_input_2000s.nc")
R2_NN_addP = file["R2_test"].mean(dim=['longitude'],skipna=True).squeeze()

R2_original["hybrid_ht"] = R2_original["hybrid_ht"]/1000
R2_addQ["hybrid_ht"] = R2_addQ["hybrid_ht"]/1000
R2_addP["hybrid_ht"] = R2_addP["hybrid_ht"]/1000
R2_zonal["hybrid_ht"] = R2_zonal["hybrid_ht"]/1000
R2_global["hybrid_ht"] = R2_global["hybrid_ht"]/1000
R2_NN["hybrid_ht"] = R2_NN["hybrid_ht"]/1000
R2_NN_addQ["hybrid_ht"] = R2_NN_addQ["hybrid_ht"]/1000
R2_NN_addP["hybrid_ht"] = R2_NN_addP["hybrid_ht"]/1000

# %% plot
vmin, vmax=0, 1.0
n_levels = 21
plt.rcParams.update({'font.size': 14})# must set in top
from read_and_use_NCL_colormaps import get_NCL_colormap
cmap = "hot_r"

fig, axes = plt.subplots(2, 3, figsize=(10, 8), layout="constrained")
cf00 = xr.plot.contourf(R2_original,ax=axes[0,0],vmin=vmin,vmax=vmax,levels=n_levels,cmap=cmap,add_colorbar=False) # cmap=cmap
axes[0,0].set_xticks(ticks=list(np.arange(-90,90,60))+[86.88])
axes[0,0].xaxis.set_minor_locator(MultipleLocator(30))

cf01 = xr.plot.contourf(R2_addQ,ax=axes[0,1],vmin=vmin,vmax=vmax,levels=n_levels,cmap=cmap,add_colorbar=False) # cmap=cmap
axes[0,1].set_xticks(ticks=list(np.arange(-90,90,60))+[86.88])
axes[0,1].xaxis.set_minor_locator(MultipleLocator(30))

cf02 = xr.plot.contourf(R2_addP,ax=axes[0,2],vmin=vmin,vmax=vmax,levels=n_levels,cmap=cmap,add_colorbar=False) # cmap=cmap
axes[0,2].set_xticks(ticks=list(np.arange(-90,90,60))+[86.88])
axes[0,2].xaxis.set_minor_locator(MultipleLocator(30))

cf10 = xr.plot.contourf(R2_NN,ax=axes[1,0],vmin=vmin,vmax=vmax,levels=n_levels,cmap=cmap,add_colorbar=False) # cmap=cmap
axes[1,0].set_xticks(ticks=list(np.arange(-90,90,60))+[86.88])
axes[1,0].xaxis.set_minor_locator(MultipleLocator(30))

cf11 = xr.plot.contourf(R2_zonal,ax=axes[1,1],vmin=vmin,vmax=vmax,levels=n_levels,cmap=cmap,add_colorbar=False) # cmap=cmap
axes[1,1].set_xticks(ticks=list(np.arange(-90,90,60))+[86.88])
axes[1,1].xaxis.set_minor_locator(MultipleLocator(30))

cf12 = xr.plot.contourf(R2_global,ax=axes[1,2],vmin=vmin,vmax=vmax,levels=n_levels,cmap=cmap,add_colorbar=False) # cmap=cmap
axes[1,2].set_xticks(ticks=list(np.arange(-90,90,60))+[86.88])
axes[1,2].xaxis.set_minor_locator(MultipleLocator(30))

clb = fig.colorbar(cf01, ax=axes[:,:], location="bottom",use_gridspec=True,fraction=0.06, pad=0.01,aspect=34,
        orientation='horizontal',ticks=np.linspace(0,1,11))   

for ax in axes[:,1]:
    ax.axes.get_yaxis().set_ticklabels([])
    ax.set_ylabel("")

for ax in axes[:,2]:
    ax.axes.get_yaxis().set_ticklabels([])
    ax.set_ylabel("")

for ax in axes[0,:]:
    ax.set_xlabel("")
    ax.set_xticklabels([])
    ax.set_ylim([2.0,50.])

for ax in axes[1,:]:
    ax.set_xlabel("")
    ax.set_xticklabels(["90S","30S","30N","90N"]) 
    ax.set_ylim([2.0,50.])

axes[0,0].set_title("(a) Original Scheme")
axes[0,1].set_title("(b) Add Q")
axes[0,2].set_title("(c) Add P")
axes[1,0].set_title("(d) NN")
axes[1,1].set_title("(e) Zonal")
axes[1,2].set_title("(f) Global")

axes[0,0].set_ylabel("Height (km)")
axes[1,0].set_ylabel("Height (km)")

plt.savefig(Outfilepath+"O3_offline_R2_compare.png", dpi=400)
plt.savefig(Outfilepath+"O3_offline_R2_compare.pdf")
plt.savefig(Outfilepath+"O3_offline_R2_compare.eps", format="eps")
plt.show()
plt.close()

# %% plot
vmin, vmax=0, 1.0
n_levels = 21
plt.rcParams.update({'font.size': 14})# must set in top
from read_and_use_NCL_colormaps import get_NCL_colormap
cmap = "hot_r"

fig, axes = plt.subplots(2, 3, figsize=(10, 8), layout="constrained")
cf00 = xr.plot.contourf(R2_original,ax=axes[0,0],vmin=vmin,vmax=vmax,levels=n_levels,cmap=cmap,add_colorbar=False) # cmap=cmap
axes[0,0].set_xticks(ticks=list(np.arange(-90,90,60))+[86.88])
axes[0,0].xaxis.set_minor_locator(MultipleLocator(30))

cf01 = xr.plot.contourf(R2_addQ,ax=axes[0,1],vmin=vmin,vmax=vmax,levels=n_levels,cmap=cmap,add_colorbar=False) # cmap=cmap
axes[0,1].set_xticks(ticks=list(np.arange(-90,90,60))+[86.88])
axes[0,1].xaxis.set_minor_locator(MultipleLocator(30))

cf02 = xr.plot.contourf(R2_addP,ax=axes[0,2],vmin=vmin,vmax=vmax,levels=n_levels,cmap=cmap,add_colorbar=False) # cmap=cmap
axes[0,2].set_xticks(ticks=list(np.arange(-90,90,60))+[86.88])
axes[0,2].xaxis.set_minor_locator(MultipleLocator(30))

cf10 = xr.plot.contourf(R2_NN,ax=axes[1,0],vmin=vmin,vmax=vmax,levels=n_levels,cmap=cmap,add_colorbar=False) # cmap=cmap
axes[1,0].set_xticks(ticks=list(np.arange(-90,90,60))+[86.88])
axes[1,0].xaxis.set_minor_locator(MultipleLocator(30))

cf11 = xr.plot.contourf(R2_NN_addQ,ax=axes[1,1],vmin=vmin,vmax=vmax,levels=n_levels,cmap=cmap,add_colorbar=False) # cmap=cmap
axes[1,1].set_xticks(ticks=list(np.arange(-90,90,60))+[86.88])
axes[1,1].xaxis.set_minor_locator(MultipleLocator(30))

cf12 = xr.plot.contourf(R2_NN_addP,ax=axes[1,2],vmin=vmin,vmax=vmax,levels=n_levels,cmap=cmap,add_colorbar=False) # cmap=cmap
axes[1,2].set_xticks(ticks=list(np.arange(-90,90,60))+[86.88])
axes[1,2].xaxis.set_minor_locator(MultipleLocator(30))

clb = fig.colorbar(cf01, ax=axes[:,:], location="bottom",use_gridspec=True,fraction=0.06, pad=0.01,aspect=34,
        orientation='horizontal',ticks=np.linspace(0,1,11))   

for ax in axes[:,1]:
    ax.axes.get_yaxis().set_ticklabels([])
    ax.set_ylabel("")

for ax in axes[:,2]:
    ax.axes.get_yaxis().set_ticklabels([])
    ax.set_ylabel("")

for ax in axes[0,:]:
    ax.set_xlabel("")
    ax.set_xticklabels([])
    ax.set_ylim([2.0,50.])

for ax in axes[1,:]:
    ax.set_xlabel("")
    ax.set_xticklabels(["90S","30S","30N","90N"]) 
    ax.set_ylim([2.0,50.])

axes[0,0].set_title("(a) Original Scheme")
axes[0,1].set_title("(b) Add Q")
axes[0,2].set_title("(c) Add P")
axes[1,0].set_title("(d) NN")
axes[1,1].set_title("(e) NN Add Q")
axes[1,2].set_title("(f) NN Add P")

axes[0,0].set_ylabel("Height (km)")
axes[1,0].set_ylabel("Height (km)")

plt.savefig(Outfilepath+"O3_offline_R2_compare1.png", dpi=400)
plt.savefig(Outfilepath+"O3_offline_R2_compare1.pdf")
plt.savefig(Outfilepath+"O3_offline_R2_compare1.eps", format="eps")
plt.show()
plt.close()


# %%
