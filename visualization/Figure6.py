# plot Figure 6 in the paper
# Long-term differences in total column ozone from mloz
# column ozone is first calculated by Figure6_column_ozone_integration.py

#%% 
import numpy as np
import xarray as xr
import cftime
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import pandas as pd

Outfilepath = "./output_column/"

if not os.path.exists(Outfilepath):
    os.makedirs(Outfilepath)
    print("The new directory ./{} is created!".format(Outfilepath))


#%% Data preparation for UKESM
Infilepath = "/home/hk-project-iconart2/ou4895/UKESM/output_column_ozone/"
# UKESM full chem
ozone_ukesm_true_pi_file = xr.open_dataset(Infilepath+'fullChem_column_ozone_piCTRL_1990_2009_UKESM_timmean.nc',decode_times=True)
Ozone_ukesm_true_pi = ozone_ukesm_true_pi_file['Column_ozone_Dobson']
ozone_ukesm_true_4co2_file = xr.open_dataset(Infilepath+'fullChem_column_ozone_4CO2_1990_2009_UKESM_timmean.nc',decode_times=True)
Ozone_ukesm_true_4co2 = ozone_ukesm_true_4co2_file['Column_ozone_Dobson']

# UKESM mloz
ozone_ukesm_onl_pi_file = xr.open_dataset(Infilepath+'mloz_column_ozone_piCTRL_2030_2049_UKESM_timmean.nc',decode_times=True)
Ozone_ukesm_onl_pi = ozone_ukesm_onl_pi_file['Column_ozone_Dobson']
ozone_ukesm_onl_4co2_file = xr.open_dataset(Infilepath+'mloz_column_ozone_4CO2_2030_2049_UKESM_timmean.nc',decode_times=True)
Ozone_ukesm_onl_4co2 = ozone_ukesm_onl_4co2_file['Column_ozone_Dobson']

# diff
Ozone_ukesm_diff_pi = (Ozone_ukesm_onl_pi - Ozone_ukesm_true_pi)/Ozone_ukesm_true_pi*100
Ozone_ukesm_diff_4co2 = (Ozone_ukesm_onl_4co2 - Ozone_ukesm_true_4co2)/Ozone_ukesm_true_4co2*100


#%% Data preparation for ICON 
# UKESM full chem (2x2 grid)
ozone_icon_true_pi_file = xr.open_dataset(Infilepath+"fullChem_column_ozone_piCTRL_1990_2009_UKESM_timmean_2x2.nc",decode_times=True)
Ozone_icon_true_pi = ozone_icon_true_pi_file['Column_ozone_Dobson']
ozone_icon_true_4co2_file = xr.open_dataset(Infilepath+'fullChem_column_ozone_4CO2_1990_2009_UKESM_timmean_2x2.nc',decode_times=True)
Ozone_icon_true_4co2 = ozone_icon_true_4co2_file['Column_ozone_Dobson']

# ICON mloz
ozone_icon_onl_pi_file = xr.open_dataset(Infilepath+'mloz_column_ozone_piCTRL_1852_1881_ICON_timmean.nc',decode_times=True)
Ozone_icon_onl_pi = ozone_icon_onl_pi_file['Column_ozone_Dobson'][::2,::2]
ozone_icon_onl_4co2_file = xr.open_dataset(Infilepath+'mloz_column_ozone_4CO2_1852_1881_ICON_timmean.nc',decode_times=True)
Ozone_icon_onl_4co2 = ozone_icon_onl_4co2_file['Column_ozone_Dobson'][::2,::2]

# diff
Ozone_icon_diff_pi = (Ozone_icon_onl_pi - Ozone_icon_true_pi)/Ozone_icon_true_pi*100
Ozone_icon_diff_4co2 = (Ozone_icon_onl_4co2 - Ozone_icon_true_4co2)/Ozone_icon_true_4co2*100


# %% Climatology plot
import cartopy.crs as ccrs
from read_and_use_NCL_colormaps import get_NCL_colormap
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker


vmin=None; vmax=None
cf_levels = 21
plt.rcParams.update({'font.size': 16})# must set in top
proj = ccrs.PlateCarree(central_longitude=180)
cmap = get_NCL_colormap('BlueWhiteOrangeRed')

# This has do be done to deal with the transition from 359°E to 1°E in this case 
# (otherwise you would end up with one non-filled longitudinal strip
Ozone_ukesm_diff_pi_wrap, lon_ukesm_wrap = add_cyclic_point(Ozone_ukesm_diff_pi, coord=Ozone_ukesm_diff_pi.longitude)
Ozone_ukesm_diff_4co2_wrap, _ = add_cyclic_point(Ozone_ukesm_diff_4co2, coord=Ozone_ukesm_diff_4co2.longitude)
Ozone_icon_diff_pi_wrap, lon_icon_wrap = add_cyclic_point(Ozone_icon_diff_pi, coord=Ozone_icon_diff_pi.lon)
Ozone_icon_diff_4co2_wrap, _ = add_cyclic_point(Ozone_icon_diff_4co2, coord=Ozone_icon_diff_4co2.lon)
Ozone_ukesm_true_pi_wrap, _ = add_cyclic_point(Ozone_ukesm_true_pi, coord=Ozone_ukesm_true_pi.longitude)
Ozone_ukesm_true_4co2_wrap, _ = add_cyclic_point(Ozone_ukesm_true_4co2, coord=Ozone_ukesm_true_4co2.longitude)
Ozone_icon_true_pi_wrap, _ = add_cyclic_point(Ozone_icon_true_pi, coord=Ozone_icon_true_pi.lon)
Ozone_icon_true_4co2_wrap, _ = add_cyclic_point(Ozone_icon_true_4co2, coord=Ozone_icon_true_4co2.lon)

contourf_ukesm_ls = [Ozone_ukesm_diff_pi_wrap,Ozone_ukesm_diff_4co2_wrap]
contourf_icon_ls = [Ozone_icon_diff_pi_wrap,Ozone_icon_diff_4co2_wrap]
contour_ukesm_ls = [Ozone_ukesm_true_pi_wrap,Ozone_ukesm_true_4co2_wrap]
contour_icon_ls = [Ozone_icon_true_pi_wrap,Ozone_icon_true_4co2_wrap]
sub_str = [["(a)","(b)"],["(c)","(d)"]]


vmin,vmax = -9,9
c_lev = 12
fig,axes = plt.subplots(2,2,figsize=(10.5, 5.25),subplot_kw={'projection':proj},layout="constrained")  # 创建画布 (x, y)

X, Y = np.meshgrid(lon_ukesm_wrap, Ozone_ukesm_true_pi.latitude)
X1, Y1 = np.meshgrid(lon_icon_wrap, Ozone_icon_true_pi.lat)
leftlon, rightlon, lowerlat, upperlat = (-180, 180, -90, 90)
lon_formatter = cticker.LongitudeFormatter()
lat_formatter = cticker.LatitudeFormatter()

for i in range(2):
    for j in range(2): 

        if i==0:
            cf1=axes[i,j].contourf(X,Y,contourf_ukesm_ls[j],vmin=vmin,vmax=vmax,extend='neither',zorder=0, levels=c_lev,
                                transform=proj, cmap=cmap)
            c1 = axes[i,j].contour(X,Y,contour_ukesm_ls[j],levels=10,cmap=None,colors='k',linewidths=0.5)
        if i==1:
            cf1=axes[i,j].contourf(X1,Y1,contourf_icon_ls[j],vmin=vmin,vmax=vmax,extend='neither',zorder=0, levels=c_lev,
                                transform=proj, cmap=cmap)
            c1 = axes[i,j].contour(X1,Y1,contour_icon_ls[j],levels=10,cmap=None,colors='k',linewidths=0.5)

        axes[i,j].clabel(c1,fontsize=10)
        axes[i,j].add_feature(cfeature.COASTLINE.with_scale('50m'), 
                            facecolor='none', edgecolor='grey', linewidth=1.)
        axes[i,j].set_extent([leftlon, rightlon, lowerlat, upperlat], crs=proj)
        axes[i,j].set_xticks(np.arange(-180, 181, 60), crs=proj)
        axes[i,j].set_yticks(np.arange(-90, 91, 30), crs=proj)
        axes[i,j].yaxis.set_major_formatter(lat_formatter)
        axes[i,j].set_xlabel("")
        axes[i,j].xaxis.set_major_formatter(lon_formatter)
        axes[i,j].tick_params(axis='both',direction='out',labelsize=13)
        axes[i,j].text(0.03,1.05,sub_str[i][j], horizontalalignment='center', verticalalignment='center', transform=axes[i,j].transAxes)

axes[0,0].set_title("piCTRL")
axes[0,1].set_title("$4\mathrm{xCO}_{2}$")
axes[0,0].set_ylabel("UKESM")
axes[1,0].set_ylabel("ICON")
axes[0,1].set_yticklabels("")
axes[1,1].set_yticklabels("")
axes[0,0].set_xticklabels("")
axes[0,1].set_xticklabels("")
axes[0,1].set_ylabel("")
axes[1,1].set_ylabel("")

clb = plt.colorbar(cf1, ax=axes[:,:], location="right",use_gridspec=True,
                   orientation='vertical',fraction=0.5, pad=0.01) # fraction=0.06,
clb.set_label('%', labelpad=5, rotation=270) #, labelpad=12, rotation=270

plt.savefig(Outfilepath+"Column_O3_online.png", dpi=400)
plt.savefig(Outfilepath+"Column_O3_online.pdf")
plt.savefig(Outfilepath+"Column_O3_online.eps")
plt.show()
plt.close()

# %%
