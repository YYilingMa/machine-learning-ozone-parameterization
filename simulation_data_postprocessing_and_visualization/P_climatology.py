# plot figures for the paper
# climatological fields
# problem: ICON is geometric height, while UKESM is relative height.

#%% 
import numpy as np
import xarray as xr
import cftime
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
# from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['`Arial`']})

import os
# from scipy.stats import spearmanr
import pandas as pd
# import nc_time_axis

Outfilepath = "./output_Fig1_origin/"

if not os.path.exists(Outfilepath):
    os.makedirs(Outfilepath)
    print("The new directory ./{} is created!".format(Outfilepath))

#%% Data preparation for UKESM (on relative height)
Infilepath = "/hkfs/work/workspace/scratch/ou4895-ukesm2/data/"
# piCTRL
ozone_true_pi_file = xr.open_dataset(Infilepath+'data_ozone_1990_2009_UKESM_timmean.nc',decode_times=True, engine="netcdf4") # 1960-1-1 to 2009-12-30
Ozone_true_pi = ozone_true_pi_file['field2101'].squeeze()*1e6/1.657 # MASS MIXING RATIO kg/kg-1 -> ppmv
# ozone_onl_pi_file = xr.open_dataset(Infilepath+'dp830a.pw2065_84_timmean.nc',
#                                     decode_times=True, engine="netcdf4")
ozone_onl_pi_file = xr.open_dataset(Infilepath+'dl969a.pw.2030_49_timmean.nc',decode_times=True)
Ozone_onl_pi = ozone_onl_pi_file['unspecified'].squeeze()*1e6/1.657 # MASS MIXING RATIO kg/kg-1  -> ppmv

# 4CO2
ozone_true_4CO2_file = xr.open_dataset(Infilepath+'data_ozone_1990_2009_4CO2_UKESM_timmean.nc',
                                       decode_times=True, engine="netcdf4") # 1960-1-1 to 2009-12-30
Ozone_true_4CO2 = ozone_true_4CO2_file['field2101_1'].squeeze()*1e6/1.657 # MASS MIXING RATIO kg/kg-1 -> ppmv
ozone_onl_4CO2_file = xr.open_dataset(Infilepath+'dl971a.pw.2030_49_timmean.nc',
                                      decode_times=True, engine="netcdf4") # 2000-1-1 to 2049-12-30
# ozone_onl_4CO2_file = xr.open_dataset(Infilepath+'do860a.pw2065_84_timmean.nc',
#                                       decode_times=True, engine="netcdf4") # 2000-1-1 to 2049-12-30
Ozone_onl_4CO2 = ozone_onl_4CO2_file['unspecified'].squeeze()*1e6/1.657 # MASS MIXING RATIO kg/kg-1  -> ppmv

# zonal mean
# oz_true_pi_zonal = Ozone_true_pi.mean(dim=['longitude'],skipna=True).squeeze()
# oz_onl_pi_zonal = Ozone_onl_pi.mean(dim=['longitude'],skipna=True).squeeze()
# oz_true_4CO2_zonal = Ozone_true_4CO2.mean(dim=['longitude'],skipna=True).squeeze()
# oz_onl_4CO2_zonal = Ozone_onl_4CO2.mean(dim=['longitude'],skipna=True).squeeze()

# O3 response to 4CO2
Ozone_response_true = Ozone_true_4CO2 - Ozone_true_pi
Ozone_response_onl = Ozone_onl_4CO2 - Ozone_onl_pi

# diff between mloz & full chem
oz_diff_pi = (Ozone_onl_pi - Ozone_true_pi)/Ozone_true_pi*100
oz_diff_4CO2 = (Ozone_onl_4CO2 - Ozone_true_4CO2)/Ozone_true_4CO2*100
oz_diff_response = (Ozone_response_onl - Ozone_response_true)
oz_true_pi_zonal = Ozone_true_pi.mean(dim=['longitude'],skipna=True).squeeze()
oz_true_4CO2_zonal = Ozone_true_4CO2.mean(dim=['longitude'],skipna=True).squeeze()
oz_onl_4CO2_zonal = Ozone_onl_4CO2.mean(dim=['longitude'],skipna=True).squeeze()
oz_diff_pi_zonal = oz_diff_pi.mean(dim=['longitude'],skipna=True).squeeze()
oz_diff_4CO2_zonal = oz_diff_4CO2.mean(dim=['longitude'],skipna=True).squeeze()
oz_diff_response_zonal = oz_diff_response.mean(dim=['longitude'],skipna=True).squeeze()

# oz_true_pi_zonal = Ozone_true_pi.squeeze()
# oz_onl_pi_zonal = Ozone_onl_pi.squeeze()
# oz_diff_pi_zonal = Ozone_diff_pi.squeeze()
# oz_true_4CO2_zonal = Ozone_true_4CO2.squeeze()
# oz_onl_4CO2_zonal = Ozone_onl_4CO2.squeeze()
# oz_diff_4CO2_zonal = Ozone_diff_4CO2.squeeze()
# oz_true_response_zonal = Ozone_response_true.squeeze()
# oz_onl_response_zonal = Ozone_response_onl.squeeze()

oz_true_pi_zonal["hybrid_ht"] = oz_true_pi_zonal["hybrid_ht"]/1000
# oz_onl_pi_zonal["hybrid_ht"] = oz_onl_pi_zonal["hybrid_ht"]/1000
oz_true_4CO2_zonal["hybrid_ht"] = oz_true_4CO2_zonal["hybrid_ht"]/1000
oz_onl_4CO2_zonal["hybrid_ht"] = oz_onl_4CO2_zonal["hybrid_ht"]/1000
oz_diff_pi_zonal["hybrid_ht"] = oz_diff_pi_zonal["hybrid_ht"]/1000
oz_diff_4CO2_zonal["hybrid_ht"] = oz_diff_4CO2_zonal["hybrid_ht"]/1000
oz_diff_response_zonal["hybrid_ht"] = oz_diff_response_zonal["hybrid_ht"]/1000
Ozone_response_true["hybrid_ht"] = Ozone_response_true["hybrid_ht"]/1000
Ozone_response_onl["hybrid_ht"] = Ozone_response_onl["hybrid_ht"]/1000

#%% Data preparation for ICON (on geometric height)
# Ozone_true is from UKESM full chem (in ppmv)
# piCTRL
mloz_pi_path = "/hkfs/work/workspace/scratch/ou4895-icon2_restored/ou4895-icon2-1748481666/amip_piCTRL_R2B5_mloz_180/"
ozone_true_pi_file1 = xr.open_dataset(Infilepath+'data_ozone_on_geometric_height_1990_2009_piCTRL_UKESM_timmean_1x1.nc',
                                      decode_times=True, engine="netcdf4")
Ozone_true_pi1 = ozone_true_pi_file1['O3'].squeeze()*1e6/1.657 # MASS MIXING RATIO kg/kg-1 -> ppmv
ozone_onl_pi_file1 = xr.open_dataset(mloz_pi_path+"var_amip_piCTRL_R2B5_mloz_180_atm_3d_hl_1852_81_timmean.nc",
                                    decode_times=True, engine="netcdf4")
Ozone_onl_pi1 = ozone_onl_pi_file1["TRO3_chemtr"].squeeze()*1e6 # ppmv

# 4CO2
mloz_4co2_path='/hkfs/work/workspace/scratch/ou4895-icon2_restored/ou4895-icon2-1748481666/amip_4CO2_R2B5_mloz_14/'
ozone_true_4co2_file1 = xr.open_dataset(Infilepath+'data_ozone_on_geometric_height_1990_2009_4CO2_UKESM_timmean_1x1.nc',
                                        decode_times=True, engine="netcdf4")
Ozone_true_4co21 = ozone_true_4co2_file1['O3'].squeeze()*1e6/1.657 # MASS MIXING RATIO kg/kg-1 -> ppmv
ozone_onl_4co2_file1 = xr.open_dataset(mloz_4co2_path+"var_amip_4CO2_R2B5_mloz_14_atm_3d_hl_1852_81_timmean.nc",
                                    decode_times=True, engine="netcdf4")
Ozone_onl_4co21 = ozone_onl_4co2_file1["TRO3_chemtr"].squeeze()*1e6 # ppmv

# zonal mean
# oz_true_pi_zonal1 = Ozone_true_pi1.mean(dim=['lon'],skipna=True).squeeze()
# oz_true_4co2_zonal1 = Ozone_true_4co21.mean(dim=['lon'],skipna=True).squeeze()
# oz_onl_pi_zonal1 = Ozone_onl_pi1.mean(dim=['lon'],skipna=True).squeeze()
# oz_onl_4CO2_zonal1 = Ozone_onl_4co21.mean(dim=['lon'],skipna=True).squeeze()

LAT = Ozone_true_pi1["lat"]
LON = Ozone_true_pi1["lon"]
H = Ozone_true_pi1["height"]
Ozone_true_pi_value = Ozone_true_pi1.values
Ozone_true_4CO2_value = Ozone_true_4co21.values
Ozone_onl_pi_value = Ozone_onl_pi1.values[::-1,:]
Ozone_onl_4CO2_value = Ozone_onl_4co21.values[::-1,:]

# O3 response to 4CO2
Ozone_response_onl1_value = Ozone_onl_4CO2_value - Ozone_onl_pi_value
Ozone_response_true1_value = Ozone_true_4CO2_value - Ozone_true_pi_value
Ozone_response_true1 = Ozone_true_4co21 - Ozone_true_pi1

# diff between mloz & full chem
Ozone_diff_pi = (Ozone_onl_pi_value - Ozone_true_pi_value)/Ozone_true_pi_value*100
Ozone_diff_4CO2 = (Ozone_onl_4CO2_value - Ozone_true_4CO2_value)/Ozone_true_4CO2_value*100
Ozone_diff_response = (Ozone_response_onl1_value - Ozone_response_true1_value)

OZ_diff_pi = xr.DataArray(
    Ozone_diff_pi, 
    dims = ['height', 'lat', 'lon'], # 维度名称
    coords = {'height':H, 'lat':LAT, 'lon':LON},
    name="ozone_pi", # 数组名称
    )
OZ_diff_4CO2 = xr.DataArray(
    Ozone_diff_4CO2,
    dims = ['height', 'lat', 'lon'], # 维度名称
    coords = {'height':H, 'lat':LAT, 'lon':LON},
    name="ozone_4CO2", # 数组名称
    )
OZ_diff_response = xr.DataArray(
    Ozone_diff_response,
    dims = ['height', 'lat', 'lon'], # 维度名称
    coords = {'height':H, 'lat':LAT, 'lon':LON},
    name="ozone_response", # 数组名称
    )

oz_true_pi_zonal1 = Ozone_true_pi1.mean(dim=['lon'],skipna=True).squeeze()
oz_true_4co2_zonal1 = Ozone_true_4co21.mean(dim=['lon'],skipna=True).squeeze()
oz_response_true_zonal1 = Ozone_response_true1.mean(dim=['lon'],skipna=True).squeeze()
# oz_onl_pi_zonal1 = Ozone_onl_pi1.mean(dim=['lon'],skipna=True).squeeze()
# oz_onl_4CO2_zonal1 = Ozone_onl_4co21.mean(dim=['lon'],skipna=True).squeeze()
oz_diff_pi_zonal1 = OZ_diff_pi.mean(dim=['lon'],skipna=True).squeeze()
oz_diff_4CO2_zonal1 = OZ_diff_4CO2.mean(dim=['lon'],skipna=True).squeeze()
oz_diff_response_zonal1 = OZ_diff_response.mean(dim=['lon'],skipna=True).squeeze()

oz_true_pi_zonal1["height"] = oz_true_pi_zonal1["height"]/1000
oz_true_4co2_zonal1["height"] = oz_true_4co2_zonal1["height"]/1000
oz_response_true_zonal1["height"] = oz_response_true_zonal1["height"]/1000
oz_diff_pi_zonal1["height"] = oz_diff_pi_zonal1["height"]/1000
oz_diff_4CO2_zonal1["height"] = oz_diff_4CO2_zonal1["height"]/1000
oz_diff_response_zonal1["height"] = oz_diff_response_zonal1["height"]/1000


# %% Climatology plot
vmin=-15; vmax=15
vmin1=-15; vmax1=15
# vmin=None; vmax=None
# vmin1=None; vmax1=None
cf_levels = 20; cf_levels1=21
plt.rcParams.update({'font.size': 16})# must set in top
fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")

from read_and_use_NCL_colormaps import get_NCL_colormap
cmap = get_NCL_colormap('BlueWhiteOrangeRed')
# cmap = "hot_r"
# from matplotlib.colors import LinearSegmentedColormap
# cmap1=LinearSegmentedColormap.from_list('', ['royalblue','blue','lightblue','white', 'yellow','orange','red'])
# cmap1 = plt.get_cmap('RdYlBu_r') # Append _r to the name of any built-in colormap to get the reversed version

cf1 = xr.plot.contourf(oz_diff_pi_zonal,ax=axes[0,0],vmin=vmin,vmax=vmax,levels=cf_levels+1,cmap=cmap,add_colorbar=False) # cmap=cmap
c1 = xr.plot.contour(oz_true_pi_zonal,ax=axes[0,0],levels=12,cmap=None,colors='k',linewidths=0.5)
axes[0,0].clabel(c1)
# ax00.set_xticks(ticks=list(np.arange(-90,90,60))+[89.375])
axes[0,0].set_xticks(ticks=list(np.arange(-90,90,60))+[89.375])
axes[0,0].xaxis.set_minor_locator(MultipleLocator(30))

cf2 = xr.plot.contourf(oz_diff_4CO2_zonal,ax=axes[0,1],vmin=vmin,vmax=vmax,levels=cf_levels+1,cmap=cmap,add_colorbar=False) # cmap=cmap
c1 = xr.plot.contour(oz_true_4CO2_zonal,ax=axes[0,1],levels=12,cmap=None,colors='k',linewidths=0.5)
axes[0,1].clabel(c1)
axes[0,1].set_xticks(ticks=list(np.arange(-90,90,60))+[89.375])
axes[0,1].xaxis.set_minor_locator(MultipleLocator(30))
clb = fig.colorbar(cf2, ax=axes[0,1], location="right",use_gridspec=True,fraction=0.06, aspect=None, pad=0.02,
              orientation='vertical')
clb.set_label('%', rotation=270) #, labelpad=12

cf3 = xr.plot.contourf(oz_diff_response_zonal,ax=axes[0,2],vmin=-1,vmax=1,levels=cf_levels1,cmap=cmap,add_colorbar=False) # cmap=cmap
c1 = xr.plot.contour(oz_response_true_zonal1,ax=axes[0,2],levels=12,cmap=None,colors='k',linewidths=0.5)
axes[0,2].clabel(c1)
axes[0,2].set_xticks(ticks=list(np.arange(-90,90,60))+[89.375])
axes[0,2].xaxis.set_minor_locator(MultipleLocator(30))
clb = fig.colorbar(cf3, ax=axes[0,2], location="right",use_gridspec=True,fraction=0.06, aspect=None, pad=0.02, 
              orientation='vertical')
clb.set_label('ppmv', rotation=270)

cf4 = xr.plot.contourf(oz_diff_pi_zonal1,ax=axes[1,0],vmin=vmin1,vmax=vmax1,levels=21,cmap=cmap,add_colorbar=False)
c1 = xr.plot.contour(oz_true_pi_zonal1,ax=axes[1,0],levels=12,cmap=None,colors='k',linewidths=0.5)
axes[1,0].clabel(c1)
axes[1,0].set_xticks(ticks=list(np.arange(-90,90,60))+[89.375])
axes[1,0].xaxis.set_minor_locator(MultipleLocator(30))

cf5 = xr.plot.contourf(oz_diff_4CO2_zonal1,ax=axes[1,1],vmin=vmin1,vmax=vmax1,levels=21,cmap=cmap,add_colorbar=False)
c1 = xr.plot.contour(oz_true_4co2_zonal1,ax=axes[1,1],levels=12,cmap=None,colors='k',linewidths=0.5)
axes[1,1].clabel(c1)
axes[1,1].set_xticks(ticks=list(np.arange(-90,90,60))+[89.375])
axes[1,1].xaxis.set_minor_locator(MultipleLocator(30))
clb = fig.colorbar(cf5, ax=axes[1,1], location="right",use_gridspec=True,fraction=0.06, aspect=None, pad=0.02,
              orientation='vertical')
clb.set_label('%', rotation=270) #, labelpad=12

cf6 = xr.plot.contourf(oz_diff_response_zonal1,ax=axes[1,2],vmin=-1,vmax=1,levels=cf_levels1,cmap=cmap,add_colorbar=False)
c1 = xr.plot.contour(oz_response_true_zonal1,ax=axes[1,2],levels=12,cmap=None,colors='k',linewidths=0.5)
axes[1,2].clabel(c1)
axes[1,2].set_xticks(ticks=list(np.arange(-90,90,60))+[89.375])
axes[1,2].xaxis.set_minor_locator(MultipleLocator(30))
clb = fig.colorbar(cf6, ax=axes[1,2], location="right",use_gridspec=True,fraction=0.06, aspect=None, pad=0.02, 
                orientation='vertical')
clb.set_label('ppmv', rotation=270)

for ax in [axes[0,1],axes[0,2],axes[1,1],axes[1,2]]:
    ax.axes.get_yaxis().set_ticklabels([])
    ax.set_ylabel("")

for ax in [axes[0,0],axes[0,1],axes[0,2]]:
    ax.axes.get_xaxis().set_ticklabels([])

for ax in [axes[0,0],axes[0,1],axes[0,2],axes[1,0],axes[1,1],axes[1,2]]:
    ax.set_xlabel("")
    ax.set_ylim([2.0,57])

for ax in [axes[1,0],axes[1,1],axes[1,2]]:
    ax.set_title("")
    ax.set_xticklabels(["90S","30S","30N","90N"])  

axes[0,0].set_title("piCTRL",fontsize=20)
axes[0,1].set_title(r"$4\mathrm{xCO}_{2}$",fontsize=20)
axes[0,2].set_title(r"$4\mathrm{xCO}_{2}$ - piCTRL",fontsize=20)
axes[0,0].set_ylabel("UKESM",fontsize=20)
axes[1,0].set_ylabel("ICON",fontsize=20)

axes[0,0].text(-87,53,"(a)")
axes[0,1].text(-87,53,"(b)")
axes[0,2].text(-87,53,"(c)")
axes[1,0].text(-87,53,"(d)")
axes[1,1].text(-87,53,"(e)")
axes[1,2].text(-87,53,"(f)")

# ax00.set_ylabel("Height(km)")
# ax10.set_ylabel("Height(km)")
# ax20.set_ylabel("Height(km)")

# norm = Normalize(vmin=vmin, vmax=vmax)
# cax = fig.add_axes([0.15, 0.04, 0.7, 0.026])
# fig.colorbar(cf1, ax=axs, cax=cax, orientation='horizontal',use_gridspec=True)

# plt.tight_layout()
plt.savefig(Outfilepath+"Climatology_O3_online.png", dpi=600)
plt.savefig(Outfilepath+"Climatology_O3_online.pdf")
plt.savefig(Outfilepath+"Climatology_O3_online.eps")
plt.show()
plt.close()


# %%
