# plot figures for the paper
# standard deviations

#%% 
import numpy as np
import xarray as xr
import cftime
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['`Helvetica`']})
# rc('text', usetex=True)
import os
# from scipy.stats import spearmanr
import pandas as pd
# import nc_time_axis

Infilepath = "/hkfs/work/workspace/scratch/ou4895-ukesm2/data/"
Outfilepath = "./output_Fig3/"

if not os.path.exists(Outfilepath):
    os.makedirs(Outfilepath)
    print("The new directory ./{} is created!".format(Outfilepath))

#%% Data preparation
# piCTRL
ozone_true_pi_file = xr.open_dataset(Infilepath+'data_ozone_1990_2009_UKESM_std.nc',decode_times=True) # 1960-1-1 to 2009-12-30
Ozone_true_pi = ozone_true_pi_file['field2101'].squeeze()*1e6/1.657
ozone_onl_pi_file = xr.open_dataset(Infilepath+'dl969a.pw.2030_49_std.nc',decode_times=True) # 2000-1-1 to 2049-12-30
Ozone_onl_pi = ozone_onl_pi_file['unspecified'].squeeze()*1e6/1.657
Ozone_diff_pi = (Ozone_onl_pi - Ozone_true_pi)

# 4CO2
ozone_true_4CO2_file = xr.open_dataset(Infilepath+'data_ozone_1990_2009_4CO2_UKESM_std.nc',decode_times=True) # 1960-1-1 to 2009-12-30
Ozone_true_4CO2 = ozone_true_4CO2_file['field2101_1'].squeeze()*1e6/1.657
ozone_onl_4CO2_file = xr.open_dataset(Infilepath+'dl971a.pw.2030_49_std.nc',decode_times=True) # 2000-1-1 to 2049-12-30
Ozone_onl_4CO2 = ozone_onl_4CO2_file['unspecified'].squeeze()*1e6/1.657
Ozone_diff_4CO2 = (Ozone_onl_4CO2 - Ozone_true_4CO2)

oz_true_pi_zonal = Ozone_true_pi.mean(dim=['longitude'],skipna=True).squeeze()
oz_onl_pi_zonal = Ozone_onl_pi.mean(dim=['longitude'],skipna=True).squeeze()
oz_diff_pi_zonal = Ozone_diff_pi.mean(dim=['longitude'],skipna=True).squeeze()
oz_true_4CO2_zonal = Ozone_true_4CO2.mean(dim=['longitude'],skipna=True).squeeze()
oz_onl_4CO2_zonal = Ozone_onl_4CO2.mean(dim=['longitude'],skipna=True).squeeze()
oz_diff_4CO2_zonal = Ozone_diff_4CO2.mean(dim=['longitude'],skipna=True).squeeze()

oz_true_pi_zonal["hybrid_ht"] = oz_true_pi_zonal["hybrid_ht"]/1000
oz_onl_pi_zonal["hybrid_ht"] = oz_onl_pi_zonal["hybrid_ht"]/1000
oz_true_4CO2_zonal["hybrid_ht"] = oz_true_4CO2_zonal["hybrid_ht"]/1000
oz_onl_4CO2_zonal["hybrid_ht"] = oz_onl_4CO2_zonal["hybrid_ht"]/1000
oz_diff_pi_zonal["hybrid_ht"] = oz_diff_pi_zonal["hybrid_ht"]/1000
oz_diff_4CO2_zonal["hybrid_ht"] = oz_diff_4CO2_zonal["hybrid_ht"]/1000

# %% std plot
vmin=-1; vmax=1
plt.rcParams.update({'font.size': 15})# must set in top
from read_and_use_NCL_colormaps import get_NCL_colormap
cmap = get_NCL_colormap('BlueWhiteOrangeRed')
# cmap = "hot_r"
# from matplotlib.colors import LinearSegmentedColormap
# cmap1=LinearSegmentedColormap.from_list('', ['royalblue','blue','lightblue','white', 'yellow','orange','red'])
# cmap1 = plt.get_cmap('RdYlBu_r') # Append _r to the name of any built-in colormap to get the reversed version
fig, axes = plt.subplots(1, 2, figsize=(10, 6), layout="constrained")
cf1 = xr.plot.contourf(oz_diff_pi_zonal,ax=axes[0],vmin=vmin,vmax=vmax,levels=25,cmap=cmap,add_colorbar=False) # cmap=cmap
c1 = xr.plot.contour(oz_true_pi_zonal,ax=axes[0],levels=9,cmap=None,colors='k',linewidths=0.5)
axes[0].clabel(c1,fontsize=14)
axes[0].set_xticks(ticks=list(np.arange(-90,90,60))+[89.375])
axes[0].xaxis.set_minor_locator(MultipleLocator(30))

cf2 = xr.plot.contourf(oz_diff_4CO2_zonal,ax=axes[1],vmin=vmin,vmax=vmax,levels=25,cmap=cmap,add_colorbar=False) # cmap=cmap
c1 = xr.plot.contour(oz_true_4CO2_zonal,ax=axes[1],levels=9,cmap=None,colors='k',linewidths=0.5)
axes[1].clabel(c1,fontsize=14)
axes[1].set_xticks(ticks=list(np.arange(-90,90,60))+[89.375])
axes[1].xaxis.set_minor_locator(MultipleLocator(30))
clb = fig.colorbar(cf2, ax=axes[1], location="right",use_gridspec=True,fraction=0.05, aspect=None, pad=0.02,
              orientation='vertical')

for ax in [axes[1]]:
    ax.axes.get_yaxis().set_ticklabels([])
    ax.set_ylabel("")

# for ax in [ax00,ax01,ax02]:
#     ax.axes.get_xaxis().set_ticklabels([])
    
for ax in [axes[0],axes[1]]:
    ax.set_xlabel("")
    ax.set_xticklabels(["90S","30S","30N","90N"]) 
    ax.set_ylim([2.0,57.])

axes[0].set_title("(a) piCTRL")
axes[1].set_title("(b) $4\mathrm{xCO}_{2}$")
axes[0].set_ylabel("Height [km]")


# ax00.set_ylabel("Height(km)")
# ax10.set_ylabel("Height(km)")
# ax20.set_ylabel("Height(km)")

# norm = Normalize(vmin=vmin, vmax=vmax)
# cax = fig.add_axes([0.15, 0.04, 0.7, 0.026])
# fig.colorbar(cf1, ax=axs, cax=cax, orientation='horizontal',use_gridspec=True)

# plt.tight_layout()
# fig.canvas.draw_idle()
plt.savefig(Outfilepath+"Std_O3_online.png", dpi=400)
plt.savefig(Outfilepath+"Std_O3_online.pdf")
plt.savefig(Outfilepath+"Std_O3_online.eps", format="eps")
plt.show()
plt.close()


# %%
