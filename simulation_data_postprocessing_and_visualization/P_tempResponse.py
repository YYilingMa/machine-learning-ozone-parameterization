# plot figures for the paper
# Temperature response profile

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

Infilepath = "/hkfs/work/workspace/scratch/ou4895-ukesm2/data/"
Outfilepath = "./output_Fig5/"

if not os.path.exists(Outfilepath):
    os.makedirs(Outfilepath)
    print("The new directory ./{} is created!".format(Outfilepath))

#%% Data preparation for left subfigure (UKESM)
True_pi_file = xr.open_dataset(Infilepath+'data_temp_1990_2009_UKESM_timmean.nc',decode_times=True) # 1960-1-1 to 2009-12-30
Temp_true_pi = True_pi_file['temp'].squeeze()
Onl_pi_file = xr.open_dataset(Infilepath+'dl969a.pw.2030_49_timmean.nc',decode_times=True) # 2000-1-1 to 2049-12-30
Temp_onl_pi = Onl_pi_file['temp'].squeeze()
True_4CO2_file = xr.open_dataset(Infilepath+'data_temp_1990_2009_4CO2_UKESM_timmean.nc',decode_times=True) # 1960-1-1 to 2009-12-30
Temp_true_4CO2 = True_4CO2_file['temp'].squeeze()
Onl_4CO2_file = xr.open_dataset(Infilepath+'dl971a.pw.2030_49_timmean.nc',decode_times=True) # 2000-1-1 to 2049-12-30
Temp_onl_4CO2 = Onl_4CO2_file['temp'].squeeze()

# temp response
temp_mloz_diff = Temp_onl_4CO2 - Temp_onl_pi
temp_true_diff = Temp_true_4CO2 - Temp_true_pi
temp_ukesm_mloz_diff_zonal = temp_mloz_diff.mean(dim=['longitude'],skipna=True)
temp_ukesm_mloz_diff_zonal["hybrid_ht"] = temp_mloz_diff["hybrid_ht"]/1000
temp_ukesm_true_diff_zonal = temp_true_diff.mean(dim=['longitude'],skipna=True)
temp_ukesm_true_diff_zonal["hybrid_ht"] = temp_true_diff["hybrid_ht"]/1000

temp_ukesm_mloz_diff_eq = temp_ukesm_mloz_diff_zonal.sel(latitude=slice(-6,6)).mean("latitude")
temp_ukesm_true_diff_eq = temp_ukesm_true_diff_zonal.sel(latitude=slice(-6,6)).mean("latitude")

#%% Data preparation for right subfigure (ICON)
mloz_path1 = '/hkfs/work/workspace/scratch/ou4895-icon2_restored/ou4895-icon2-1748481666/amip_piCTRL_R2B5_mloz_180/'
mloz_path2 = '/hkfs/work/workspace/scratch/ou4895-icon2_restored/ou4895-icon2-1748481666/amip_4CO2_R2B5_mloz_14/'
mloz_pi_file = xr.open_dataset(mloz_path1+"var_amip_piCTRL_R2B5_mloz_180_atm_3d_hl_1852_81_timmean.nc")
mloz_4co2_file = xr.open_dataset(mloz_path2+"var_amip_4CO2_R2B5_mloz_14_atm_3d_hl_1852_81_timmean.nc")

prescribe_path1 = '/hkfs/work/workspace/scratch/ou4895-icon2_restored/ou4895-icon2-1748481666/amip_piCTRL_R2B5_ukesmSST_O3_51/'
prescribe_path2 = '/hkfs/work/workspace/scratch/ou4895-icon2_restored/ou4895-icon2-1748481666/amip_4CO2_R2B5_ukesmSST_O3_7/' # prescribed by piCTRL o3
prescribe_path3 = '/hkfs/work/workspace/scratch/ou4895-icon2_restored/ou4895-icon2-1748481666/amip_4CO2_R2B5_ukesmSST_O3_6/' # prescribed by 4CO2 o3
prescribe_pi_file = xr.open_dataset(prescribe_path1+"var_amip_piCTRL_R2B5_ukesmSST_O3_51_atm_3d_hl_1852_81_timmean.nc")
# prescribed-ukesm-pi o3
prescribe_4co2_piO3_file = xr.open_dataset(prescribe_path2+"var_amip_4CO2_R2B5_ukesmSST_O3_7_atm_3d_hl_1852_81_timmean.nc")
# prescribed-ukesm-4co2 o3  
prescribe_4co2_4co2O3_file = xr.open_dataset(prescribe_path3+"var_amip_4CO2_R2B5_ukesmSST_O3_6_atm_3d_hl_1852_81_timmean.nc")        


# temp response of non-interactive ozone vs. interactive ozone
temp_prescribe_pi_mean = prescribe_pi_file["temp"]
temp_prescribe_4co2_piO3_mean = prescribe_4co2_piO3_file["temp"]
temp_prescribe_4co2_4co2O3_mean = prescribe_4co2_4co2O3_file["temp"]
temp_prescribe_piO3_diff = temp_prescribe_4co2_piO3_mean - temp_prescribe_pi_mean
temp_prescribe_4co2O3_diff = temp_prescribe_4co2_4co2O3_mean - temp_prescribe_pi_mean
temp_prescribe_piO3_diff_zonal = temp_prescribe_piO3_diff.mean(dim=['lon'],skipna=True).squeeze()
temp_prescribe_4co2O3_diff_zonal = temp_prescribe_4co2O3_diff.mean(dim=['lon'],skipna=True).squeeze()
temp_mloz_pi_mean = mloz_pi_file["temp"]
temp_mloz_4co2_mean = mloz_4co2_file["temp"]
temp_mloz_diff = temp_mloz_4co2_mean - temp_mloz_pi_mean
temp_icon_mloz_diff_zonal = temp_mloz_diff.mean(dim=['lon'],skipna=True).squeeze()

temp_prescribe_piO3_diff_zonal["alt"] = temp_prescribe_piO3_diff["alt"]/1000
temp_prescribe_4co2O3_diff_zonal["alt"] = temp_prescribe_4co2O3_diff["alt"]/1000
temp_icon_mloz_diff_zonal["alt"] = temp_mloz_diff["alt"]/1000

temp_prescribe_piO3_diff_eq = temp_prescribe_piO3_diff_zonal.sel(lat=slice(-6,6)).mean("lat")
temp_prescribe_4co2O3_diff_eq = temp_prescribe_4co2O3_diff_zonal.sel(lat=slice(-6,6)).mean("lat")
temp_icon_mloz_diff_eq = temp_icon_mloz_diff_zonal.sel(lat=slice(-6,6)).mean("lat")

#%%
plt.rcParams.update({'font.size': 19})# must set in top
fig, axes = plt.subplots(1, 2, figsize=(10, 8), layout="constrained")

c1, = axes[0].plot(temp_ukesm_true_diff_eq,temp_ukesm_true_diff_zonal["hybrid_ht"][:],color='k',linestyle='--',linewidth=2.5) #,label="full chem"
c2, = axes[0].plot(temp_ukesm_mloz_diff_eq,temp_ukesm_mloz_diff_zonal["hybrid_ht"][:],color='r',linewidth=2.5) # ,label="mloz"
# axes[0].set_xlabel("Temp response[K]")
# axes[0].set_xlabel('Temp response to 4CO2[K]')  
axes[0].set_ylabel("Height [km]")
axes[0].set_title("(a) UKESM")
axes[0].legend(["full chem","mloz"],loc='upper right')

c3, = axes[1].plot(temp_prescribe_piO3_diff_eq,temp_prescribe_piO3_diff_eq.alt[:],linewidth=2.5,color='k',linestyle='-')
c4, = axes[1].plot(temp_prescribe_4co2O3_diff_eq,temp_prescribe_4co2O3_diff_zonal.alt[:],linewidth=2.5,color='k',linestyle='--')
c5, = axes[1].plot(temp_icon_mloz_diff_eq,temp_icon_mloz_diff_zonal.alt[:],color='r')
axes[1].set_yticklabels([])
axes[1].set_title("(b) ICON")
axes[1].legend(["prescribe-pi $\mathrm{O}_{3}$","prescribe-$4\mathrm{xCO}_{2}$ $\mathrm{O}_{3}$","mloz"],loc='upper right')

for ax in axes[:]:
    ax.set_xticks(ticks=list(np.arange(-30,21,10)))
    ax.set_yticks(ticks=list(np.arange(0,61,10)))
    ax.set_ylim([0,60])

fig.supxlabel('Temp response to $4\mathrm{xCO}_{2}$ [K]')
plt.savefig(Outfilepath+"Temp_response_online.png", dpi=400)
plt.savefig(Outfilepath+"Temp_response_online.pdf")
plt.savefig(Outfilepath+"Temp_response_online.eps")

# %%
