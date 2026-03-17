# plot Figure 2 in the paper
# Time series on chosen points

#%% 
import numpy as np
import xarray as xr
import cftime
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
import os
import pandas as pd

Infilepath = "/hkfs/work/workspace/scratch/ou4895-ukesm2/data/"
Outfilepath = "./output_TimeSeries/"

if not os.path.exists(Outfilepath):
    os.makedirs(Outfilepath)
    print("The new directory ./{} is created!".format(Outfilepath))

#%% Data preparation
h_sel = [0,20000,26100,41800]
lat_sel = [-64,0,40]
lon_sel = [50,100,300]
# piCTRL
ozone_true_pi_file = xr.open_dataset(Infilepath+'fullChem_ozone_piCTRL_1962_2034_UKESM_monmean.nc',
                                     decode_times=True).sel(hybrid_ht=h_sel,latitude=lat_sel,longitude=lon_sel,method="nearest") # 
Ozone_true_pi = ozone_true_pi_file['field2101']*1e6/1.657
ozone_onl_pi_file = xr.open_dataset(Infilepath+'mloz_ozone_piCTRL_2000_2049_UKESM_monmean.nc',
                                    decode_times=True).sel(hybrid_ht=h_sel,latitude=lat_sel,longitude=lon_sel,method="nearest") # 2000-1-1 to 2049-12-30
Ozone_onl_pi = ozone_onl_pi_file['unspecified']*1e6/1.657 # Ozone output from online mloz run

# 4CO2
ozone_true_4CO2_file = xr.open_dataset(Infilepath+'fullChem_ozone_4CO2_1962_2034_UKESM_monmean.nc',
                                       decode_times=True).sel(hybrid_ht=h_sel,latitude=lat_sel,longitude=lon_sel,method="nearest") # 1960-1-1 to 2009-12-30
Ozone_true_4CO2 = ozone_true_4CO2_file['field2101_1']*1e6/1.657
ozone_onl_4CO2_file = xr.open_dataset(Infilepath+'mloz_ozone_4CO2_2000_2049_UKESM_monmean.nc',
                                      decode_times=True).sel(hybrid_ht=h_sel,latitude=lat_sel,longitude=lon_sel,method="nearest") # need to be changed into last 25-yr data 
Ozone_onl_4CO2 = ozone_onl_4CO2_file['unspecified']*1e6/1.657 # Ozone output from online mloz run

Ozone_true_pi["t"] = Ozone_onl_pi["t"]
Ozone_true_4CO2["t"] = Ozone_onl_4CO2["t"]

Ozone_clim_pi = Ozone_true_pi.groupby("t.month").mean('t')

#%% select points
Ozone_true_sel = [
    Ozone_true_pi.sel(hybrid_ht=41800,longitude=100,latitude=40,method="nearest"),
    Ozone_true_pi.sel(hybrid_ht=26100,longitude=0,latitude=0,method="nearest"), 
    Ozone_true_pi.sel(hybrid_ht=20000,longitude=300,latitude=0,method="nearest"),
    Ozone_true_pi.sel(hybrid_ht=0,longitude=150,latitude=-84,method="nearest"), #
    Ozone_true_4CO2.sel(hybrid_ht=41800,longitude=100,latitude=40,method="nearest"),
    Ozone_true_4CO2.sel(hybrid_ht=26100,longitude=0,latitude=0,method="nearest"),
    Ozone_true_4CO2.sel(hybrid_ht=20000,longitude=300,latitude=0,method="nearest"),
    Ozone_true_4CO2.sel(hybrid_ht=0,longitude=150,latitude=-84,method="nearest"),
]
Ozone_onl_sel = [
    Ozone_onl_pi.sel(hybrid_ht=41800,longitude=100,latitude=40,method="nearest"),
    Ozone_onl_pi.sel(hybrid_ht=26100,longitude=0,latitude=0,method="nearest"),
    Ozone_onl_pi.sel(hybrid_ht=20000,longitude=300,latitude=0,method="nearest"),
    Ozone_onl_pi.sel(hybrid_ht=0,longitude=150,latitude=-84,method="nearest"),
    Ozone_onl_4CO2.sel(hybrid_ht=41800,longitude=100,latitude=40,method="nearest"),
    Ozone_onl_4CO2.sel(hybrid_ht=26100,longitude=0,latitude=0,method="nearest"),
    Ozone_onl_4CO2.sel(hybrid_ht=20000,longitude=300,latitude=0,method="nearest"),
    Ozone_onl_4CO2.sel(hybrid_ht=0,longitude=150,latitude=-84,method="nearest"),
]
Ozone_clim_pi_sel = [
    Ozone_clim_pi.sel(hybrid_ht=41800,longitude=100,latitude=40,method="nearest"),
    Ozone_clim_pi.sel(hybrid_ht=26100,longitude=0,latitude=0,method="nearest"),
    Ozone_clim_pi.sel(hybrid_ht=20000,longitude=300,latitude=0,method="nearest"),
    Ozone_clim_pi.sel(hybrid_ht=0,longitude=150,latitude=-84,method="nearest"),
    Ozone_clim_pi.sel(hybrid_ht=41800,longitude=100,latitude=40,method="nearest"),
    Ozone_clim_pi.sel(hybrid_ht=26100,longitude=0,latitude=0,method="nearest"),
    Ozone_clim_pi.sel(hybrid_ht=20000,longitude=300,latitude=0,method="nearest"),
    Ozone_clim_pi.sel(hybrid_ht=0,longitude=150,latitude=-84,method="nearest")
]

#%% plot
alp_list = ["(a) piCTRL;","(b) piCTRL;","(c) piCTRL;","(d) piCTRL;",r"(e) $4\mathrm{xCO}_{2}$;",r"(f) $4\mathrm{xCO}_{2}$;",r"(g) $4\mathrm{xCO}_{2}$;",r"(h) $4\mathrm{xCO}_{2}$;"]
plt.rcParams.update({'font.size': 13})
fig, axes = plt.subplots(8, 1, sharex=True, figsize=(10, 14), layout="constrained")
for i in range(8):
    alt_sel = np.array(Ozone_true_sel[i]["hybrid_ht"]/1000).round(1)
    lat_sel = np.array(Ozone_true_sel[i]["latitude"]).round(1)
    lon_sel = np.array(Ozone_true_sel[i]["longitude"]).round(1)
    axes[i].plot(np.tile(Ozone_clim_pi_sel[i],50),'grey',linestyle="-")
    axes[i].plot(Ozone_true_sel[i],color="k",label="full chem",linestyle="-")
    axes[i].plot(Ozone_onl_sel[i],color="r",label="mloz",linestyle="-")
    titname = alp_list[i]+" alt:"+str(alt_sel)+"km, lat:"+str(lat_sel)+", lon: "+str(lon_sel)+"E"
    axes[i].set_title(titname,loc="left")
    axes[i].set_xticks(ticks=list(np.arange(0,601,60)))
    axes[i].set_xlim([12,600])

axes[7].set_xticklabels(np.arange(0,51,5))
axes[7].set_xlabel("Simulation year",fontsize=14)
# fig.text(-0.02, 0.5, 'Ozone volume mixing ratio [ppmv]', va='center', rotation='vertical')
axes[3].set_ylabel('Ozone volume mixing ratio [ppmv]',y=0,fontsize=14)
plt.legend(loc='best', bbox_to_anchor=(1.01, -.16),ncols=2) # (x,y)
plt.savefig(Outfilepath+"TimeSeries_O3_UKESM.png", dpi=600)
plt.savefig(Outfilepath+"TimeSeries_O3_UKESM.pdf")
plt.savefig(Outfilepath+"TimeSeries_O3_UKESM.eps")
plt.show()
plt.close()

# %%
