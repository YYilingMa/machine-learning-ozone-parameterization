# plot figures for the paper
# PDF distributions

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
# from scipy.stats import spearmanr
# import pandas as pd
from read_and_use_NCL_colormaps import get_NCL_colormap
# import nc_time_axis

Infilepath = "/hkfs/work/workspace/scratch/ou4895-ukesm2/data/"
Outfilepath = "./output_FigPDF_bandwidth0.25/"

if not os.path.exists(Outfilepath):
    os.makedirs(Outfilepath)
    print("The new directory ./{} is created!".format(Outfilepath))

#%% Data preparation
h_sel = [0,20000,26100,41800]
lat_sel = [-64,0,40]
lon_sel = [50,100,300]

# UKESM full chem
ukesm_pi_path = "/hkfs/work/workspace/scratch/ou4895-ukesm2/data/"
# ukesm_pi_file = xr.open_dataset(ukesm_pi_path+"data_ozone_1990_2009_UKESM_2x2.nc",
#                 decode_times=True).sel(hybrid_ht=h_sel,lat=lat_sel,lon=lon_sel,method="nearest") # UKESM full chem, piCTRL
ukesm_pi_file = xr.open_dataset(ukesm_pi_path+"data_ozone_1990_2009_UKESM.nc",
                decode_times=True, engine="netcdf4").sel(hybrid_ht=h_sel,latitude=lat_sel,longitude=lon_sel,method="nearest") # UKESM full chem, piCTRL
o3_pi_ukesm_true = ukesm_pi_file["field2101"]/1.657*1e6 # ppmv
ukesm_4co2_path = "/hkfs/work/workspace/scratch/ou4895-ukesm2/data/"
# ukesm_4co2_file = xr.open_dataset(ukesm_pi_path+"data_ozone_1990_2009_4CO2_UKESM_2x2.nc",
#                 decode_times=True).sel(hybrid_ht=h_sel,lat=lat_sel,lon=lon_sel,method="nearest") # UKESM full chem, 4CO2
ukesm_4co2_file = xr.open_dataset(ukesm_pi_path+"data_ozone_1990_2009_4CO2_UKESM.nc",
                decode_times=True).sel(hybrid_ht=h_sel,latitude=lat_sel,longitude=lon_sel,method="nearest") # UKESM full chem, 4CO2
o3_4co2_ukesm_true = ukesm_4co2_file["field2101_1"]/1.657*1e6 # ppmv

# UKESM mloz
# ozone_onl_pi_file = xr.open_dataset(Infilepath+'dl969_ozone_a.pw.2030_49_2x2.nc',
#                                     decode_times=True).sel(hybrid_ht=h_sel,lat=lat_sel,lon=lon_sel,method="nearest") # 2000-1-1 to 2049-12-30
ozone_onl_pi_file = xr.open_dataset(Infilepath+'dl969_ozone_a.pw.2030_49.nc',
                                    decode_times=True).sel(hybrid_ht=h_sel,latitude=lat_sel,longitude=lon_sel,method="nearest") # 2000-1-1 to 2049-12-30
oz_pi_ukesm_mloz = ozone_onl_pi_file['unspecified'].squeeze()*1e6/1.657
# ozone_onl_4CO2_file = xr.open_dataset(Infilepath+'dl971_ozone_a.pw.2030_49_2x2.nc',
#                                       decode_times=True).sel(hybrid_ht=h_sel,lat=lat_sel,lon=lon_sel,method="nearest") # 2000-1-1 to 2049-12-30
ozone_onl_4CO2_file = xr.open_dataset(Infilepath+'dl971_ozone_a.pw.2030_49.nc',
                                      decode_times=True).sel(hybrid_ht=h_sel,latitude=lat_sel,longitude=lon_sel,method="nearest") # 2000-1-1 to 2049-12-30
oz_4co2_ukesm_mloz = ozone_onl_4CO2_file['unspecified'].squeeze()*1e6/1.657

# # ICON mloz
# mloz_pi_path = "/hkfs/work/workspace/scratch/ou4895-icon2/amip_piCTRL_R2B5_mloz_180/"
# mloz_pi_file = xr.open_dataset(mloz_pi_path+"O3_amip_piCTRL_R2B5_mloz_180_atm_3d_hl_1852_81.nc",
#                                 decode_times=True).sel(alt=[0,18000,26100,41800],lat=[-90,3,88],lon=[20,58,180],method="nearest")
# o3_pi_icon_mloz = mloz_pi_file["TRO3_chemtr"]*1e6 # ppmv
# mloz_4co2_path='/hkfs/work/workspace/scratch/ou4895-icon2/amip_4CO2_R2B5_mloz_14/'
# mloz_4co2_file = xr.open_dataset(mloz_4co2_path+"O3_amip_4CO2_R2B5_mloz_14_atm_3d_hl_1852_81.nc",
#                                 decode_times=True).sel(alt=[0,18000,26100,41800],lat=[-90,3,88],lon=[20,58,180],method="nearest")
# o3_4co2_icon_mloz = mloz_4co2_file["TRO3_chemtr"]*1e6 # ppmv

#!! ukesm 4CO2 run is not prescribed by 10/20 years mean SSTs
#!! so its pdf distribution is not fully comparable with mloz!
#!! and their height coordinates are also different under 17km!

#%% pdf distributions
oz_pi_ukesm_true_ls = [
                o3_pi_ukesm_true.sel(hybrid_ht=41800,longitude=100,latitude=40,method="nearest"),
                o3_pi_ukesm_true.sel(hybrid_ht=26100,longitude=0,latitude=0,method="nearest"),
                o3_pi_ukesm_true.sel(hybrid_ht=20000,longitude=300,latitude=0,method="nearest"),
                o3_pi_ukesm_true.sel(hybrid_ht=0,longitude=100,latitude=-84,method="nearest"),]
oz_4co2_ukesm_true_ls = [
                o3_4co2_ukesm_true.sel(hybrid_ht=41800,longitude=100,latitude=40,method="nearest"),
                o3_4co2_ukesm_true.sel(hybrid_ht=26100,longitude=0,latitude=0,method="nearest"),
                o3_4co2_ukesm_true.sel(hybrid_ht=20000,longitude=300,latitude=0,method="nearest"),
                o3_4co2_ukesm_true.sel(hybrid_ht=0,longitude=100,latitude=-84,method="nearest"),]
oz_pi_ukesm_mloz_ls = [
                oz_pi_ukesm_mloz.sel(hybrid_ht=41800,longitude=100,latitude=40,method="nearest"),
                oz_pi_ukesm_mloz.sel(hybrid_ht=26100,longitude=0,latitude=0,method="nearest"),
                oz_pi_ukesm_mloz.sel(hybrid_ht=20000,longitude=300,latitude=0,method="nearest"),
                oz_pi_ukesm_mloz.sel(hybrid_ht=0,longitude=100,latitude=-84,method="nearest"),]
oz_4co2_ukesm_mloz_ls = [
                oz_4co2_ukesm_mloz.sel(hybrid_ht=41800,longitude=100,latitude=40,method="nearest"),
                oz_4co2_ukesm_mloz.sel(hybrid_ht=26100,longitude=0,latitude=0,method="nearest"),
                oz_4co2_ukesm_mloz.sel(hybrid_ht=20000,longitude=300,latitude=0,method="nearest"),
                oz_4co2_ukesm_mloz.sel(hybrid_ht=0,longitude=100,latitude=-84,method="nearest"),]
# oz_pi_icon_mloz_ls = [
#                 o3_pi_icon_mloz.sel(alt=41800,lon=0,lat=0,method="nearest"),
#                 o3_pi_icon_mloz.sel(alt=26100,lon=0,lat=0,method="nearest"),
#                 o3_pi_icon_mloz.sel(alt=18000,lon=58,lat=88,method="nearest"),
#                 o3_pi_icon_mloz.sel(alt=0,lon=180,lat=-90,method="nearest"),]
# oz_4co2_icon_mloz_ls = [
#                 o3_4co2_icon_mloz.sel(alt=41800,lon=0,lat=0,method="nearest"),
#                 o3_4co2_icon_mloz.sel(alt=26100,lon=0,lat=0,method="nearest"),
#                 o3_4co2_icon_mloz.sel(alt=18000,lon=58,lat=88,method="nearest"),
#                 o3_4co2_icon_mloz.sel(alt=0,lon=180,lat=-90,method="nearest"),]

ht_plot = [np.round(oz_pi_ukesm_true_ls[i].hybrid_ht.values/1000,2) for i in range(4)]
lon_plot = [np.round(oz_pi_ukesm_true_ls[i].longitude.values,1) for i in range(4)]
lat_plot = [np.round(oz_pi_ukesm_true_ls[i].latitude.values,1) for i in range(4)]
catergory_ls = [
    "Alt:{:.1f}km, lat:{:.1f}, lon:{:.1f}E".format(ht_plot[0],lat_plot[0],lon_plot[0]), 
    "Alt:{:.1f}km, lat:{:.1f}, lon:{:.1f}E".format(ht_plot[1],lat_plot[1],lon_plot[1]),
    "Alt:{:.1f}km, lat:{:.1f}, lon:{:.1f}E".format(ht_plot[2],lat_plot[2],lon_plot[2]),
    "Alt:{:.1f}km, lat:{:.1f}, lon:{:.1f}E".format(ht_plot[3],lat_plot[3],lon_plot[3])]

#%%
from sklearn.neighbors import KernelDensity
logprob_pi_ukesm_true=[];logprob_4co2_ukesm_true=[];logprob_pi_ukesm_mloz=[]
logprob_pi_icon_mloz=[];logprob_4co2_icon_mloz=[];logprob_4co2_ukesm_mloz=[]
for i in range(len(catergory_ls)):
    if i<=1: 
        x_d = np.linspace(0, 15, 500)
    elif i==2:
        x_d = np.linspace(0, 6, 500)
    elif i==3:
        x_d = np.linspace(0, 0.1, 500)

    bdwidth = oz_pi_ukesm_true_ls[i].mean().values * 0.02
    x_pi_ukesm_true = oz_pi_ukesm_true_ls[i].values.reshape(-1,1)
    x_4co2_ukesm_true = oz_4co2_ukesm_true_ls[i].values.reshape(-1,1)
    x_pi_ukesm_mloz = oz_pi_ukesm_mloz_ls[i].values.reshape(-1,1)
    x_4co2_ukesm_mloz = oz_4co2_ukesm_mloz_ls[i].values.reshape(-1,1)
    # x_pi_icon_mloz = oz_pi_icon_mloz_ls[i].values.reshape(-1,1)
    # x_4co2_icon_mloz = oz_4co2_icon_mloz_ls[i].values.reshape(-1,1)

    kde_pi_ukesm_true = KernelDensity(kernel='gaussian', bandwidth=bdwidth).fit(x_pi_ukesm_true)
    kde_4co2_ukesm_true = KernelDensity(kernel='gaussian', bandwidth=bdwidth).fit(x_4co2_ukesm_true)
    kde_pi_ukesm_mloz = KernelDensity(kernel='gaussian', bandwidth=bdwidth).fit(x_pi_ukesm_mloz)
    kde_4co2_ukesm_mloz = KernelDensity(kernel='gaussian', bandwidth=bdwidth).fit(x_4co2_ukesm_mloz)
    # kde_pi_icon_mloz = KernelDensity(kernel='gaussian', bandwidth=0.25).fit(x_pi_icon_mloz)
    # kde_4co2_icon_mloz = KernelDensity(kernel='gaussian', bandwidth=0.25).fit(x_4co2_icon_mloz)

    # score_samples returns the log of the probability density
    logprob_pi_ukesm_true.append(kde_pi_ukesm_true.score_samples(x_d[:, None]))
    logprob_4co2_ukesm_true.append(kde_4co2_ukesm_true.score_samples(x_d[:, None]))
    logprob_pi_ukesm_mloz.append(kde_pi_ukesm_mloz.score_samples(x_d[:, None]))
    logprob_4co2_ukesm_mloz.append(kde_4co2_ukesm_mloz.score_samples(x_d[:, None]))
    # logprob_pi_icon_mloz.append(kde_pi_icon_mloz.score_samples(x_d[:, None]))
    # logprob_4co2_icon_mloz.append(kde_4co2_icon_mloz.score_samples(x_d[:, None]))


# %% pdf plot
sub_str = ["(a)","(b)","(c)","(d)"]
plt.rcParams.update({'font.size': 14})# must set in top ,'font.family':"Helvetica"
# plt.rcParams["fontname"] = ["Arial"]
# from matplotlib.colors import LinearSegmentedColormap
# cmap1=LinearSegmentedColormap.from_list('', ['royalblue','blue','lightblue','white', 'yellow','orange','red'])
# cmap1 = plt.get_cmap('RdYlBu_r') # Append _r to the name of any built-in colormap to get the reversed version

fig, axes = plt.subplots(2, 2, figsize=(7, 6)) # layout="constrained"

for i in range(2):
    for j in range(2):
        if i==0: 
            x_d = np.linspace(0, 15, 500)
        elif j==0:
            x_d = np.linspace(0, 6, 500)
        elif j==1:
            x_d = np.linspace(0, 0.1, 500)

        c1, = axes[i,j].plot(x_d, np.exp(logprob_pi_ukesm_true[i*2+j]), color="k", label='full chem;pi')
        c2, = axes[i,j].plot(x_d, np.exp(logprob_pi_ukesm_mloz[i*2+j]), color="r", label='mloz;pi')
        # c3, = axes[i,j].plot(x_d, np.exp(logprob_pi_icon_mloz[i*2+j]), color="b", label='icon mloz;pi')
        c4, = axes[i,j].plot(x_d, np.exp(logprob_4co2_ukesm_true[i*2+j]), color="k",linestyle='--',label='full chem;$4\mathrm{xCO}_{2}$')
        c5, = axes[i,j].plot(x_d, np.exp(logprob_4co2_ukesm_mloz[i*2+j]), color="r",linestyle='--',label='mloz;$4\mathrm{xCO}_{2}$')
        # c6, = axes[i,j].plot(x_d, np.exp(logprob_4co2_icon_mloz[i*2+j]), color="b",linestyle='--',label='icon mloz;4CO2')
        axes[i,j].text(0.08,0.92,sub_str[i*2+j], horizontalalignment='center', verticalalignment='center', transform=axes[i,j].transAxes)
        axes[i,j].set_title(catergory_ls[i*2+j],fontsize=14)

for ax in axes[:,1]:
    ax.set_ylabel("")
    
for ax in axes[0,:]:
    ax.set_xlabel("")

for ax in axes[1,:]:
    ax.set_xlabel("Ozone conc [ppmv]")

axes[0,0].set_xlim([0,12])
axes[0,1].set_xlim([0,11])
axes[1,0].set_xlim([0,3])
axes[1,1].set_xlim([0,0.03])
# axes[1,1].set_ylim([0,2])
fig.legend(handles=[c1, c2, c4, c5], loc='lower center', bbox_to_anchor=(0.63, 0.0),ncol=2)
fig.tight_layout(rect=[0, 0.1, 1, 1]) # Leave space at the bottom # Calling tight_layout will turn off constrained layout!

plt.savefig(Outfilepath+"Pdf_O3_online.png", dpi=400)
plt.savefig(Outfilepath+"Pdf_O3_online.pdf")
plt.savefig(Outfilepath+"Pdf_O3_online.eps")
plt.show()
plt.close()


# %%
