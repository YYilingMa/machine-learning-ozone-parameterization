
# Temperature bias between online daily_nean_temp and training set

#%% 
import numpy as np
import xarray as xr
import cftime
import matplotlib.pyplot as plt
from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['`Helvetica`']})
# rc('text', usetex=True)
from matplotlib.ticker import MultipleLocator
import os
import pandas as pd

Infilepath = "/hkfs/work/workspace/scratch/ou4895-ukesm/data/"
Outfilepath = "./output_tempBias/"
lon_sel = np.arange(0,192,5)
lat_int = 2

if not os.path.exists(Outfilepath):
    os.makedirs(Outfilepath)
    print("The new directory ./{} is created!".format(Outfilepath))

#%% Data preparation for left subfigure (UKESM)
Onlinefilepath = "/hkfs/work/workspace/scratch/ou4895-ukesm/data/"
True_pi_file = xr.open_dataset(Infilepath+'data_temp_1990_2009_UKESM_timmean.nc',decode_times=True) # 1960-1-1 to 2009-12-30
Temp_true_pi = True_pi_file['temp'].isel(longitude=lon_sel).sel(hybrid_ht=slice(0,50000)).squeeze()
Onl_pi_file = xr.open_dataset(Infilepath+'dl969a.pw2004_07_daily_mean_temp.nc',decode_times=True) # 2000-1-1 to 2049-12-30
Temp_onl_pi0 = Onl_pi_file['unspecified'].isel(longitude=lon_sel).sel(hybrid_ht=slice(0,50000)).squeeze()
Temp_onl_pi=Temp_onl_pi0.where(Temp_onl_pi0>100,drop=True).mean("t")
# True_4CO2_file = xr.open_dataset(Infilepath+'data_temp_1990_2009_4CO2_UKESM_timmean.nc',decode_times=True) # 1960-1-1 to 2009-12-30
# Temp_true_4CO2 = True_4CO2_file['temp'].isel(longitude=lon_sel).squeeze()
# Onl_4CO2_file = xr.open_dataset(Infilepath+'dl971a.pw2004_07_daily_mean_temp.nc',decode_times=True) # 2000-1-1 to 2049-12-30
# Temp_onl_4CO20 = Onl_4CO2_file['unspecified'].isel(longitude=lon_sel).squeeze()
# Temp_onl_4CO2=Temp_onl_4CO20.where(Temp_onl_4CO20>100,drop=True).mean("t")

# temp response
temp_diff = Temp_onl_pi - Temp_true_pi
temp_diff["hybrid_ht"] = Temp_onl_pi0["hybrid_ht"]/1000
Temp_onl_pi["hybrid_ht"] = Temp_onl_pi0["hybrid_ht"]/1000
Temp_true_pi["hybrid_ht"] = Temp_onl_pi0["hybrid_ht"]/1000

#%% plot online temp versus offline temp
vmin,vmax = 190,310
cf_levels = 17
plt.rcParams.update({'font.size': 18})# must set in top
fig, axes = plt.subplots(1, 2, figsize=(10, 8), layout="constrained")

cf1 = xr.plot.contourf(Temp_onl_pi,ax=axes[0],vmin=vmin,vmax=vmax,levels=cf_levels,cmap="jet",extend="both",add_colorbar=False)
cf2 = xr.plot.contourf(Temp_true_pi,ax=axes[1],vmin=vmin,vmax=vmax,levels=cf_levels,cmap="jet",extend="both",add_colorbar=False)

clb = fig.colorbar(cf2, ax=axes[:], location="right",use_gridspec=True,fraction=0.06, aspect=None, pad=0.02,
              orientation='vertical')
axes[0].set_title("online daily_mean_temp",fontsize=20)
axes[1].set_title("true T",fontsize=20)
plt.savefig(Outfilepath+"Temp_online_offline.png", dpi=400)


# %%
from read_and_use_NCL_colormaps import get_NCL_colormap
cmap = get_NCL_colormap('BlueWhiteOrangeRed')

vmin,vmax = None,None
cf_levels = 17
plt.rcParams.update({'font.size': 18})# must set in top
fig, ax = plt.subplots(1, 1, figsize=(8, 8), layout="constrained")

cf = xr.plot.contourf(temp_diff.mean("longitude"),ax=ax,vmin=vmin,vmax=vmax,levels=cf_levels,cmap=cmap,extend="both",add_colorbar=False)

clb = fig.colorbar(cf, ax=ax, location="right",use_gridspec=True,fraction=0.06, aspect=None, pad=0.02,
              orientation='vertical')
ax.set_title("")
ax.set_xticks(ticks=list(np.arange(-90,90,60))+[89.375])
ax.set_xticklabels(["90S","30S","30N","90N"])  
ax.set_ylabel("Height(km)")
ax.set_xlabel("")

plt.savefig(Outfilepath+"Temp_diff_online_offline.png", dpi=400)
plt.savefig(Outfilepath+"Temp_diff_online_offline.pdf")
plt.savefig(Outfilepath+"Temp_diff_online_offline.eps")

# %%
