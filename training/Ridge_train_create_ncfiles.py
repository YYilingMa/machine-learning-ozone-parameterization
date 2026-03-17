# This script creates ncfiles for Ridge coefs and y_scaler.

#%%
import joblib 
import xarray as xr
import numpy as np
import netCDF4

str_exp = "piCTRL" # piCTRL or 4CO2

#%%
n_level = 76
n_lat = 144
n_lon = 192
n_batch = 48
n_proma = 192//48
y_means = np.empty((n_level,n_lat,n_lon))
y_scales = np.empty((n_level,n_lat,n_lon))
x_means = np.empty((n_level,n_lat,n_lon))
x_scales = np.empty((n_level,n_lat,n_lon))

filepath = '/hkfs/work/workspace/scratch/ou4895-ukesm/output/RidgeModel_38yrsTrain_under50km_piCTRL_76lev/'
# coefs dims: (c=76 predictor levels, z=76 output levels, lat=144, lon=192)
# coefs[c, z, lat, lon] = weight connecting temperature at level c to ozone at level z
coefs = np.zeros((n_level,n_level,n_lat,n_lon))
alphas = np.zeros((n_level,n_lat,n_lon))

for i_batch in range(0,n_batch,1):
    # x = temperature (predictor), y = ozone (target)
    # mean and scale are the StandardScaler parameters used to normalize/denormalize
    ### scaler_x
    scaler_x = joblib.load(filepath+'Scaler_x_'+str_exp+'_lon_batch'+str(i_batch)+'.pkl')
    x_scale = scaler_x.scale_.reshape((n_level,n_lat,n_proma)) 
    x_mean = scaler_x.mean_.reshape((n_level,n_lat,n_proma)) 
    x_means[:,:,i_batch*n_proma:(i_batch+1)*n_proma] = x_mean
    x_scales[:,:,i_batch*n_proma:(i_batch+1)*n_proma] = x_scale

    ### scaler_y
    scaler_y = joblib.load(filepath+'Scaler_y_'+str_exp+'_lon_batch'+str(i_batch)+'.pkl')
    y_scale = scaler_y.scale_.reshape((n_level,n_lat,n_proma)) 
    y_mean = scaler_y.mean_.reshape((n_level,n_lat,n_proma)) 
    y_means[:,:,i_batch*n_proma:(i_batch+1)*n_proma] = y_mean
    y_scales[:,:,i_batch*n_proma:(i_batch+1)*n_proma] = y_scale

    ### coefs
    models = joblib.load(filepath+'RidgeRegressor_'+str_exp+'_lon_batch'+str(i_batch)+'.pkl')
    for lati in range(0,n_lat):
        for levi in range(0,n_level):
            for j in range(0,n_proma):
                coefs[:,levi,lati,i_batch*n_proma+j] = models['lon_'+str(j)+'_lat'+str(lati)+'_level'+str(levi)].coef_
                alphas[levi,lati,i_batch*n_proma+j] = models['lon_'+str(j)+'_lat'+str(lati)+'_level'+str(levi)].alpha

#%% write out to ncfiles
### scaler_x
file_to_write = netCDF4.Dataset(filepath+'Scaler_x_40yrs_'+str_exp+'_64_sum.nc','w',dtype=np.float64,format='NETCDF4_CLASSIC')
z_dim = file_to_write.createDimension('z',n_level)
lat_dim = file_to_write.createDimension('lat',n_lat)
lon_dim = file_to_write.createDimension('lon',n_lon)
x_mean_w = file_to_write.createVariable('x_mean',np.float64,('z','lat','lon'))
x_mean_w[:] = x_means
x_scale_w = file_to_write.createVariable('x_scale',np.float64,('z','lat','lon'))
x_scale_w[:] = x_scales
file_to_write.close()

### scaler_y
file_to_write = netCDF4.Dataset(filepath+'Scaler_y_40yrs_'+str_exp+'_64_sum.nc','w',dtype=np.float64,format='NETCDF4_CLASSIC')
z_dim = file_to_write.createDimension('z',n_level)
lat_dim = file_to_write.createDimension('lat',n_lat)
lon_dim = file_to_write.createDimension('lon',n_lon)
y_mean_w = file_to_write.createVariable('y_mean',np.float64,('z','lat','lon'))
y_mean_w[:] = y_means
y_scale_w = file_to_write.createVariable('y_scale',np.float64,('z','lat','lon'))
y_scale_w[:] = y_scales
file_to_write.close()

### coefs
file_to_write = netCDF4.Dataset(filepath+'coefs_40years_'+str_exp+'_64_sum.nc','w',dtype=np.float64,format='NETCDF4_CLASSIC')
c_dim = file_to_write.createDimension('c',n_level)
z_dim = file_to_write.createDimension('z',n_level)
lat_dim = file_to_write.createDimension('lat',n_lat)
lon_dim = file_to_write.createDimension('lon',n_lon)
coefs_w = file_to_write.createVariable('coefs',np.float64,('c','z','lat','lon'))
coefs_w[:] = coefs
file_to_write.close()

### alphas
file_to_write = netCDF4.Dataset(filepath+'alphas_40years_'+str_exp+'_64_sum.nc','w',dtype=np.float64,format='NETCDF4_CLASSIC')
z_dim = file_to_write.createDimension('z',n_level)
lat_dim = file_to_write.createDimension('lat',n_lat)
lon_dim = file_to_write.createDimension('lon',n_lon)
alphas_w = file_to_write.createVariable('alphas',np.float64,('z','lat','lon'))
alphas_w[:] = alphas
file_to_write.close()

# %%
