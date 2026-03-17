#%%
import netCDF4
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as clrs
import numpy as np
import sys
import scipy.stats
import os
import xarray as xr

plt.close("all")
plt.clf()
plt.rc('font', family='sans-serif', serif='cm10')                                       
sys.stdout.flush()

Outfilepath = "./output_column_ozone/"

if not os.path.exists(Outfilepath):
    os.makedirs(Outfilepath)
    print("The new directory ./{} is created!".format(Outfilepath))

def Calculate_column_tracer(vmr_tracer,pres_tracer,temp_tracer,height_intv):
   # input: [t,height,lat,lon]
   NA = 6.02214e23
   R = 8.31446
   # Number density (molecules/m3) from ideal gas law: n = VMR * P * NA / (R * T)
   # where NA = Avogadro constant (6.022e23 mol-1), R = gas constant (8.314 J mol-1 K-1)
   ndensity = vmr_tracer*pres_tracer*NA/(R*temp_tracer)
   # Skip the lowest interface level; nconc is defined on n_h-1 layer midpoints
   nconc = ndensity[:,1:,:,:]
   n_t, n_h, n_lat, n_lon = np.shape(nconc)

   col_oz = np.zeros((n_t,n_lat,n_lon))
   for i in range(n_h):
      col_oz[:,:,:] += nconc[:,i,:,:].squeeze()*height_intv[i]
      # col_oz[:,0,:,:] is the total column ozone
   return col_oz # molecules/m2; divide by 2.69e20 to convert to Dobson Units (1 DU = 2.69e20 molecules/m2)

#%% UKESM full chem
### data preparation
Infilepath = "/hkfs/work/workspace/scratch/ou4895-ukesm/data/"
file = xr.open_dataset(Infilepath+'fullChem_piCTRL_1990_2009_UKESM_monmean.nc',decode_times=True).sel(hybrid_ht=slice(0,7.5e04))
# Convert ozone from mass mixing ratio (MMR, kg/kg) to volume mixing ratio (VMR, mol/mol)
# Factor = molecular weight of air (28.97) / molecular weight of O3 (47.998) ≈ 1/1.657
ozone = file['field2101']/1.657 # MMR -> VMR
temp = file['temp']
p = file['p']

### calculate column ozone (piCTRL)
height_intv = np.array(ozone["hybrid_ht"][1:])-np.array(ozone["hybrid_ht"][:-1])
col_oz = Calculate_column_tracer(np.array(ozone),np.array(p),np.array(temp),height_intv)
# mean over time, in Dobson unit
col_oz_ave = np.mean(col_oz[:,:,:],axis=0,dtype=np.float64).squeeze()/2.69e20 

### write out to ncfiles
LONGITUDE = file['longitude']
LATITUDE = file['latitude']

### coefs
Column = xr.DataArray(
    col_oz_ave, # ml_ozone(z,time,lat,lon)
    dims = ['latitude', 'longitude'], # dimension names
    coords = {'latitude':LATITUDE, 'longitude':LONGITUDE},
    name="Column_ozone_Dobson", # array name
    )
Column.to_netcdf(Outfilepath+"/fullChem_column_ozone_piCTRL_1990_2009_UKESM_timmean.nc","w")

# then interpolate UKESM full chem column ozone ave onto 2x2 latxlon grid using cdo


#%% UKESM mloz
file = xr.open_dataset(Infilepath+'mloz_piCTRL_2030_2049_UKESM_monmean.nc',decode_times=True).sel(hybrid_ht=slice(0,7.5e04))
file1 = xr.open_dataset(Infilepath+"mloz_ozone_piCTRL_2030_2049_UKESM_monmean.nc",decode_times=True).sel(hybrid_ht=slice(0,7.5e04))

ozone = file1['unspecified']/1.657 # MMR -> VMR
temp = file['temp']
p = file['p']

### calculate column ozone (piCTRL)
height_intv = np.array(ozone["hybrid_ht"][1:])-np.array(ozone["hybrid_ht"][:-1])
col_oz = Calculate_column_tracer(np.array(ozone),np.array(p),np.array(temp),height_intv)
# mean over time, in Dobson unit
col_oz_ave = np.mean(col_oz[:,:,:],axis=0,dtype=np.float64).squeeze()/2.69e20 

### write out to ncfiles
LONGITUDE = file['longitude']
LATITUDE = file['latitude']
Column = xr.DataArray(
    col_oz_ave, # ml_ozone(z,time,lat,lon)
    dims = ['latitude', 'longitude'], # dimension names
    coords = {'latitude':LATITUDE, 'longitude':LONGITUDE},
    name="Column_ozone_Dobson", # array name
    )
Column.to_netcdf(Outfilepath+"/mloz_column_ozone_piCTRL_2030_2049_UKESM_timmean.nc","w")

#%% ICON
### data preparation
Infilepath = "/hkfs/work/workspace/scratch/ou4895-icon2/amip_piCTRL_R2B5_mloz_180/"
file = xr.open_dataset(Infilepath+'mloz_piCTRL_1852_1881_ICON_monmean.nc',decode_times=True).isel(alt=slice(None, None, -1)) # ICON altitude coordinate is top-down; reverse to bottom-up for height integration
ozone = file['TRO3_chemtr'] # VMR
temp = file['temp']
p = file['pres']

### calculate column ozone (piCTRL)
height_intv = np.array(ozone["alt"][1:])-np.array(ozone["alt"][:-1])
col_oz = Calculate_column_tracer(np.array(ozone),np.array(p),np.array(temp),height_intv)
# mean over time, in Dobson unit
col_oz_ave = np.mean(col_oz[:,:,:],axis=0,dtype=np.float64).squeeze()/2.69e20 

### write out to ncfiles
LONGITUDE = file['lon']
LATITUDE = file['lat']
Column = xr.DataArray(
    col_oz_ave, # ml_ozone(z,time,lat,lon)
    dims = ['lat', 'lon'], # dimension names
    coords = {'lat':LATITUDE, 'lon':LONGITUDE},
    name="Column_ozone_Dobson", # array name
    )
Column.to_netcdf(Outfilepath+"/mloz_column_ozone_piCTRL_1852_1881_ICON_timmean.nc","w")

#%%
