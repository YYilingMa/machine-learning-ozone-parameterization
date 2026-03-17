# plot Figure 4 in the manuscript
# climatological fields of ozone & ozone response to 4xCO2
# from mloz prediction online in UKESM and ICON

#%% 
import numpy as np
import xarray as xr
import cftime
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import pandas as pd

Outfilepath = "./output_Climatology1/"

if not os.path.exists(Outfilepath):
    os.makedirs(Outfilepath)
    print("The new directory ./{} is created!".format(Outfilepath))

#%% Data preparation
### UKESM
ukesm_filepath = "/hkfs/work/workspace/scratch/ou4895-ukesm_restored/ou4895-ukesm2-1761661744/ou4895-ukesm2-1755999483/data/"
# piCTRL
ukesm_true_pi_file = xr.open_dataset(ukesm_filepath+'fullChem_ozone_piCTRL_1990_2009_UKESM_timmean.nc',decode_times=True)
ukesm_true_pi = ukesm_true_pi_file['field2101'].squeeze()*1e6/1.657 # MASS MIXING RATIO kg/kg-1 -> ppmv
ukesm_mloz_pi_file = xr.open_dataset(ukesm_filepath+'mloz_ozone_piCTRL_2030_2049_UKESM_timmean.nc',decode_times=True)
ukesm_mloz_pi = ukesm_mloz_pi_file['unspecified'].squeeze()*1e6/1.657 # MASS MIXING RATIO kg/kg-1  -> ppmv
# 4CO2
ukesm_true_4CO2_file = xr.open_dataset(ukesm_filepath+'fullChem_ozone_4CO2_1990_2009_UKESM_timmean.nc',decode_times=True)
ukesm_true_4CO2 = ukesm_true_4CO2_file['field2101_1'].squeeze()*1e6/1.657 # MASS MIXING RATIO kg/kg-1 -> ppmv
ukesm_mloz_4CO2_file = xr.open_dataset(ukesm_filepath+'mloz_ozone_4CO2_2030_2049_UKESM_timmean.nc',decode_times=True)
ukesm_mloz_4CO2 = ukesm_mloz_4CO2_file['unspecified'].squeeze()*1e6/1.657 # MASS MIXING RATIO kg/kg-1  -> ppmv
# O3 response to 4CO2
ukesm_true_response = (ukesm_true_4CO2 - ukesm_true_pi)/ukesm_true_pi*100
ukesm_mloz_response = (ukesm_mloz_4CO2 - ukesm_mloz_pi)/ukesm_mloz_pi*100

### ICON
# piCTRL
icon_mloz_pi_path = "/hkfs/work/workspace/scratch/ou4895-icon2_restored/ou4895-icon2-1748481666/amip_piCTRL_R2B5_mloz_180/"
icon_mloz_pi_file = xr.open_dataset(icon_mloz_pi_path+"mloz_ozone_piCTRL_1852_1881_ICON_timmean.nc",
                                    decode_times=True)
icon_mloz_pi = icon_mloz_pi_file["TRO3_chemtr"].squeeze()*1e6 # ppmv

icon_linoz_pi_path = "/hkfs/work/workspace/scratch/ou4895-icon2_restored/ou4895-icon2-1748481666/amip_piCTL_R2B5_linozO3_43/"
icon_linoz_pi_file = xr.open_dataset(icon_linoz_pi_path+"o3_amip_piCTL_R2B5_linozO3_43_atm_3d_hl_1852_81_timmean.nc",
                                    decode_times=True)
icon_linoz_pi = icon_linoz_pi_file["TRO3_chemtr"].squeeze()*1e6 # ppmv
# 4CO2
icon_mloz_4co2_path='/hkfs/work/workspace/scratch/ou4895-icon2_restored/ou4895-icon2-1748481666/amip_4CO2_R2B5_mloz_14/'
icon_mloz_4co2_file = xr.open_dataset(icon_mloz_4co2_path+"mloz_ozone_4CO2_1852_1881_ICON_timmean.nc",
                                    decode_times=True)
icon_mloz_4CO2 = icon_mloz_4co2_file["TRO3_chemtr"].squeeze()*1e6 # ppmv

icon_linoz_4co2_path = "/hkfs/work/workspace/scratch/ou4895-icon2_restored/ou4895-icon2-1748481666/amip_4CO2_R2B5_linozO3_4/"
icon_linoz_4co2_file = xr.open_dataset(icon_linoz_4co2_path+"o3_amip_4CO2_R2B5_linozO3_4_atm_3d_hl_1852_81_timmean.nc",
                                    decode_times=True)
icon_linoz_4CO2 = icon_linoz_4co2_file["TRO3_chemtr"].squeeze()*1e6 # ppmv
# O3 response to 4CO2
icon_mloz_response = (icon_mloz_4CO2 - icon_mloz_pi)/icon_mloz_pi*100
icon_linoz_response = (icon_linoz_4CO2 - icon_linoz_pi)/icon_linoz_pi*100

# %% Climatology plot
ukesm_true_pi_zonal = ukesm_true_pi.mean(dim=['longitude'],skipna=True).squeeze()
ukesm_mloz_pi_zonal = ukesm_mloz_pi.mean(dim=['longitude'],skipna=True).squeeze()
icon_mloz_pi_zonal = icon_mloz_pi.mean(dim=['lon'],skipna=True).squeeze()
icon_linoz_pi_zonal = icon_linoz_pi.mean(dim=['lon'],skipna=True).squeeze()
ukesm_true_response_zonal = ukesm_true_response.mean(dim=['longitude'],skipna=True).squeeze()
ukesm_mloz_response_zonal = ukesm_mloz_response.mean(dim=['longitude'],skipna=True).squeeze()
icon_mloz_response_zonal = icon_mloz_response.mean(dim=['lon'],skipna=True).squeeze()
icon_linoz_response_zonal = icon_linoz_response.mean(dim=['lon'],skipna=True).squeeze()

ukesm_true_pi_zonal["hybrid_ht"] = ukesm_true_pi_zonal["hybrid_ht"]/1000
ukesm_mloz_pi_zonal["hybrid_ht"] = ukesm_mloz_pi_zonal["hybrid_ht"]/1000
icon_mloz_pi_zonal["alt"] = icon_mloz_pi_zonal["alt"]/1000
icon_linoz_pi_zonal["alt"] = icon_linoz_pi_zonal["alt"]/1000
ukesm_true_response_zonal["hybrid_ht"] = ukesm_true_response_zonal["hybrid_ht"]/1000
ukesm_mloz_response_zonal["hybrid_ht"] = ukesm_mloz_response_zonal["hybrid_ht"]/1000
icon_mloz_response_zonal["alt"] = icon_mloz_response_zonal["alt"]/1000
icon_linoz_response_zonal["alt"] = icon_linoz_response_zonal["alt"]/1000

#%%
from read_and_use_NCL_colormaps import get_NCL_colormap
cmap1 = get_NCL_colormap('BlueWhiteOrangeRed')
cmap = "hot_r"
plt.rcParams.update({'font.size': 18})
figures_contourf = [[ukesm_true_pi_zonal,ukesm_mloz_pi_zonal],
                [icon_mloz_pi_zonal,icon_linoz_pi_zonal]]
figures_contour = [[ukesm_true_response_zonal,ukesm_mloz_response_zonal],
                [icon_mloz_response_zonal,icon_linoz_response_zonal]]

linoz_levels = np.arange(-80, 161, 20)
linoz_levels_label = np.array([-60,-20,0,20,80,120])
other_levels_label = np.array([-60,-20,0,20])

sub_str=[["(a)","(b)"],["(c)","(d)"]]
vmin, vmax = 0, 13
fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout="constrained")

for i in range(2):
    for j in range(2):
        cf = xr.plot.contourf(figures_contourf[i][j],ax=axes[i,j],vmin=vmin,vmax=vmax,levels=22,
                    cmap=cmap,add_colorbar=False)
        if i==1 & j==1:
            c = xr.plot.contour(figures_contour[i][j],ax=axes[i,j],levels=linoz_levels,
                                cmap=None,colors='k',linewidths=1.5)
            axes[i,j].clabel(c,inline_spacing=1,levels=linoz_levels_label,fontsize=15)
        else:        
            c = xr.plot.contour(figures_contour[i][j],ax=axes[i,j],levels=8,cmap=None,colors='k',linewidths=1.5)
            axes[i,j].clabel(c,inline_spacing=1,levels=other_levels_label,fontsize=15)

        axes[i,j].set_xticks(ticks=list(np.arange(-90,90,60))+[89.375])
        axes[i,j].xaxis.set_minor_locator(MultipleLocator(30))
        axes[i,j].set_ylim([0.5,70])
        axes[i,j].set_xlabel("")

        if j!=0:
            axes[i,j].set_yticklabels("")
        if i!=1: 
            axes[i,j].axes.get_xaxis().set_ticklabels([])
        else:
            axes[i,j].set_xticklabels(["90S","30S","30N","90N"])

clb = fig.colorbar(cf, ax=axes[:,:], location="right",use_gridspec=True,fraction=0.04, aspect=None, pad=0.005, 
              orientation='vertical')
clb.set_label('ppmv', rotation=270, labelpad=12)

axes[0,0].set_ylabel("Height (km)")
axes[1,0].set_ylabel("Height (km)")
axes[0,1].set_ylabel("")
axes[1,1].set_ylabel("")
axes[0,0].set_title("(a) UKESM full chem",fontsize=20)
axes[0,1].set_title("(b) UKESM mloz",fontsize=20)
axes[1,0].set_title("(c) ICON mloz",fontsize=20)
axes[1,1].set_title("(d) ICON Linoz",fontsize=20)

plt.savefig(Outfilepath+"Climatology_O3_online1.png", dpi=500)
plt.savefig(Outfilepath+"Climatology_O3_online1.pdf")
plt.savefig(Outfilepath+"Climatology_O3_online1.eps")
plt.show()
plt.close()

# %%
