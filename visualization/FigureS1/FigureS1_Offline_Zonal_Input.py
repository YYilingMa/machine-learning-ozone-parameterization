# use cross-section (Lat-lev) temp data to predict ozone on selected longitudes
# As comparison with prediction made with single column input, taking zonal field temp as input
# Used for Figure S1 in the supporting information

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sys
sys.setrecursionlimit(1000000)                                                                                                                                                                  
import os
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
import xarray as xr
np.set_printoptions(precision=4, suppress=True)

add_feature1 = None # ADD which fearture: hum, p, LW, SW or None
add_feature2 = None # ADD which fearture: hum, p, LW, SW or None
add_feature3 = None # ADD which fearture: hum, p, LW, SW or None
nt_train = 360*40 # training set length
nt_test = 360*10
lat_int = 3
lon_sel = [0.9375, 57.1875, 113.4375, 169.6875, 225.9375, 282.1875, 338.4375]
Outfilepath = "output_Zonal_Input_ZonalR2/"
Infilepath = "/hkfs/work/workspace/scratch/ou4895-ukesm_restored/ou4895-ukesm2-1761661744/ou4895-ukesm2-1755999483/data/"

if add_feature1 != None:
    if add_feature2 != None:
        if add_feature3 != None:
            print("Using Temp+{}+{}+{} as input features!".format(add_feature1,add_feature2,add_feature3))
        else:
            print("Using Temp+{}+{} as input features!".format(add_feature1,add_feature2))
    else:
        print("Using Temp+{} as input features!".format(add_feature1))
else:
    print("Using only Temp as input feature!")

if not os.path.exists(Outfilepath):
    os.makedirs(Outfilepath)
    print("The new directory ./{} is created!".format(Outfilepath))

def R2_score(x_pred,x_true):
    # Best possible score is 1.0, lower values are worse.
    u = ((x_pred-x_true)**2).sum(axis=0)
    v = ((x_true-x_true.mean(axis=0))**2).sum(axis=0)
    return 1-u/v

#%%
### Ozone data
ih_sel_o3 = np.concatenate((np.arange(10,36,6),np.arange(36,76,2),np.arange(76,80,4)))
ozone_file = xr.open_dataset(Infilepath+'fullChem_ozone_piCTRL_1962_2011_UKESM.nc',decode_times=True).sel(longitude=lon_sel,method="nearest") # 1960-1-1 to 1979-12-30
Ozone_train = ozone_file['field2101'][1:nt_train,ih_sel_o3,::lat_int,:]*1e6/1.657 # MASS MIXING RATIO kg/kg-1 -> ppmv
Ozone_test = ozone_file['field2101'][1-nt_test:,ih_sel_o3,::lat_int,:]*1e6/1.657 # MASS MIXING RATIO kg/kg-1 -> ppmv

oz_train_ls = []; oz_test_ls = []
for i in range(len(lon_sel)):
    oz_train_ls.append(Ozone_train.sel(longitude=lon_sel[i], method="nearest"))
    oz_test_ls.append(Ozone_test.sel(longitude=lon_sel[i], method="nearest"))

### Temp data
ih_sel_tem = np.concatenate((np.arange(10,36,4),np.arange(36,75,1),np.arange(75,80,2)))
temp_file = xr.open_dataset(Infilepath+'fullChem_temp_piCTRL_1962_2011_UKESM.nc',decode_times=False).sel(longitude=lon_sel,method="nearest") # 1960-1-1 to 2009-12-30
Temp_train=temp_file['temp'][:nt_train-1,ih_sel_tem,::lat_int,:]
Temp_test=temp_file['temp'][-nt_test:-1,ih_sel_tem,::lat_int,:]

# use cross-section Temp (lat-lev) to predict ozone on single point
tp_train_ls = []; tp_test_ls = []
for i in range(len(lon_sel)):
    tp_train_ls.append(Temp_train.sel(longitude=lon_sel[i], method="nearest"))
    tp_test_ls.append(Temp_test.sel(longitude=lon_sel[i], method="nearest"))

### Add_feature data
if add_feature1 != None:
    add_file1 = xr.open_dataset(Infilepath+'/fullChem_'+add_feature1+'_piCTRL_1962_2011_UKESM.nc',decode_times=False)
    if add_feature1=="hum": # ADD which fearture: hum, p, LW, SW
        valname = 'q'
    elif add_feature1=="p":
        valname = 'p'
    elif add_feature1=="LW":
        valname = 'lwhr'    
    elif add_feature1=="SW":
        valname = 'swhr'
    add_train1 = add_file1[valname][:nt_train-1,ih_sel_tem,::lat_int,:].sel(longitude=lon_sel,method="nearest")
    add_test1 = add_file1[valname][-nt_test:-1,ih_sel_tem,::lat_int,:].sel(longitude=lon_sel,method="nearest")

    add_train1_ls = []; add_test1_ls = []
    for i in range(len(lon_sel)):
        add_train1_ls.append(add_train1.sel(longitude=lon_sel[i], method="nearest"))
        add_test1_ls.append(add_test1.sel(longitude=lon_sel[i], method="nearest"))  
        
    if add_feature2 != None:
        add_file2 = xr.open_dataset(Infilepath+'/fullChem_'+add_feature2+'_piCTRL_1962_2011_UKESM.nc',decode_times=False)
        if add_feature2=="hum": # ADD which fearture: hum, p, LW, SW
            valname = 'q'
        elif add_feature2=="p":
            valname = 'p'
        elif add_feature2=="LW":
            valname = 'lwhr'    
        elif add_feature2=="SW":
            valname = 'swhr'
        add_train2 = add_file2[valname][:nt_train-1,ih_sel_tem,::lat_int,:].sel(longitude=lon_sel,method="nearest")
        add_test2 = add_file2[valname][-nt_test:-1,ih_sel_tem,::lat_int,:].sel(longitude=lon_sel,method="nearest")

        add_train2_ls = []; add_test2_ls = []
        for i in range(len(lon_sel)):
            add_train2_ls.append(add_train2.sel(longitude=lon_sel[i], method="nearest"))
            add_test2_ls.append(add_test2.sel(longitude=lon_sel[i], method="nearest"))  

        if add_feature3 != None:
            add_file3 = xr.open_dataset('./fullChem_'+add_feature3+'_piCTRL_1962_2011_UKESM.nc',decode_times=False)
            if add_feature3=="hum": # ADD which fearture: hum, p, LW, SW
                valname = 'q'
            elif add_feature3=="p":
                valname = 'p'
            elif add_feature3=="LW":
                valname = 'lwhr'    
            elif add_feature3=="SW":
                valname = 'swhr'
            add_train3 = add_file3[valname][:nt_train-1,ih_sel_tem,::lat_int,:].sel(longitude=lon_sel,method="nearest")
            add_test3 = add_file3[valname][-nt_test:-1,ih_sel_tem,::lat_int,:].sel(longitude=lon_sel,method="nearest")

            # use cross-section Temp (lat-lev) to predict ozone on single point
            add_train3_ls = []; add_test3_ls = []
            for i in range(len(lon_sel)):
                add_train3_ls.append(add_train3.sel(longitude=lon_sel[i], method="nearest"))
                add_test3_ls.append(add_test3.sel(longitude=lon_sel[i], method="nearest"))  
        
#%% Training and Testing
print("Using cross-section data as input")
alpha_i = [1e-12,3e-12,1e-11,3e-11,1e-10,3e-10,1e-9,3e-9,1e-8,3e-8,1e-7,3e-7,1e-6,3e-6,0.00001,0.00003,0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100,300,1000,3000,10000,30000,100000,300000,1000000,3000000,10000000,30000000,100000000]
parameters = {
    'alpha': alpha_i,
    'fit_intercept': [True],
    'max_iter':[1000],
    'random_state':[100] #If int, random_state is the seed used by the random number generator.
     # If None, the random number generator is the RandomState instance used by np.random. 
             }

for i in range(len(lon_sel)):

    best_alpha_array = []
    best_coef_array = []
    y_pred_train_array = []
    y_pred_test_array = []
    y_true_train_array = []
    y_true_test_array = []

    cv_obj = KFold(n_splits=5) # Each fold is then used once as a validation while the k - 1 remaining folds form the training set.

    X_train = np.array(tp_train_ls[i]).reshape((nt_train-1,-1))
    X_test = np.array(tp_test_ls[i]).reshape((nt_test-1,-1))
    Y_train = np.array(oz_train_ls[i]).reshape((nt_train-1,-1))
    Y_test = np.array(oz_test_ls[i]).reshape((nt_test-1,-1))
    n_t_test, n_h, n_lat = oz_test_ls[i].shape

    if add_feature1 != None:
        X_train1 = np.array(add_train1_ls[i]).reshape((nt_train-1,-1))
        X_test1 = np.array(add_test1_ls[i]).reshape((nt_test-1,-1))
        X_train = np.concatenate((X_train,X_train1), axis=1)
        X_test = np.concatenate((X_test,X_test1), axis=1)
        if add_feature2 != None:
            X_train2 = np.array(add_train2_ls[i]).reshape((nt_train-1,-1))
            X_test2 = np.array(add_test2_ls[i]).reshape((nt_test-1,-1))
            X_train = np.concatenate((X_train,X_train2), axis=1)
            X_test = np.concatenate((X_test,X_test2), axis=1)            
            if add_feature3 != None:
                X_train3 = np.array(add_train3_ls[i]).reshape((nt_train-1,-1))
                X_test3 = np.array(add_test3_ls[i]).reshape((nt_test-1,-1))
                X_train = np.concatenate((X_train,X_train3), axis=1)
                X_test = np.concatenate((X_test,X_test3), axis=1)  

    ### Standardize
    scaler=StandardScaler()
    scaler.fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)

    scaler_y=StandardScaler()
    scaler_y.fit(Y_train) # normalize output to make sure that lower tropospheric ozone concentrations are not deemed less important in the fit
    Y_train_std = scaler_y.transform(Y_train)
    Y_test_std = scaler_y.transform(Y_test)

    #### now do dimensionality reduction using PCA
    pca = PCA(n_components=0.99,copy=False)
    # use copy=False to save half of the memory
    # If False, data passed to fit are overwritten and running fit(X).transform(X)
    # will not yield the expected results, use fit_transform(X) instead.
    X_train_pca = pca.fit_transform(X_train_std) 
    X_test_pca = pca.transform(X_test_std)
    del X_train_std, X_test_std, X_train, X_test # clear memory 

    # tunning
    regr_ijk = GridSearchCV(Ridge(),parameters,cv=cv_obj,n_jobs=-1,refit=True)
    # refit : bool, str, or callable, default=True; Refit an estimator using the best found parameters on the whole dataset.
    # n_jobs: Number of jobs to run in parallel.
    # refit:  Refit an estimator using the best found parameters on the whole dataset.
    # cv: Determines the cross-validation splitting strategy.

    n_dim = n_h*n_lat
    for j in range(n_dim):

        x_train = X_train_pca; y_train = Y_train_std[:,j].reshape(-1, 1)
        x_test = X_test_pca; y_test = Y_test_std[:,j].reshape(-1, 1)

        regr_ijk.fit(x_train,y_train)

        best_alpha_array.append(regr_ijk.best_estimator_.alpha)
        best_coef_array.append(regr_ijk.best_estimator_.coef_)

        y_pred_train = regr_ijk.best_estimator_.predict(x_train)
        y_pred_test = regr_ijk.best_estimator_.predict(x_test) # predict ozone with the best alpha

        y_pred_train_array.append(y_pred_train)
        y_pred_test_array.append(y_pred_test)
        y_true_train_array.append(y_train)
        y_true_test_array.append(y_test)

    ozone_pred_train = np.array(y_pred_train_array).squeeze().T # ozone_pred_train:(time,n_dim)
    ozone_pred_test = np.array(y_pred_test_array).squeeze().T
    ozone_true_train = np.array(y_true_train_array).squeeze().T
    ozone_true_test = np.array(y_true_test_array).squeeze().T
    best_alphas = np.array(best_alpha_array)

    R2_train_section = R2_score(ozone_pred_train,ozone_true_train).reshape((n_h,n_lat)) #(height,lat)   
    R2_test_section = R2_score(ozone_pred_test,ozone_true_test).reshape((n_h,n_lat))    

    # inversed normalization
    predicted_ozone_r = scaler_y.inverse_transform(ozone_pred_test)
    Predicted_ozone = predicted_ozone_r.reshape(n_t_test,n_h,n_lat)
    True_ozone = oz_test_ls[i]

    Ozone_pred = xr.DataArray(
    Predicted_ozone,
    dims = True_ozone.dims,
    coords = True_ozone.coords,
    name="Ozone_predicted_by_Ridge"
    ) 

    R2_test = xr.DataArray(
    R2_test_section,
    dims = ['hybrid_ht', 'latitude'],
    coords = True_ozone[-1].coords,
    name="R2_test"
    ) 

    if i == 0:
        OZONE_PRED_TEST = Ozone_pred
        OZONE_TRUE_TEST = True_ozone
        R2_TEST = R2_test
    else:
        OZONE_PRED_TEST = xr.concat((OZONE_PRED_TEST,Ozone_pred),dim='longitude')
        OZONE_TRUE_TEST = xr.concat((OZONE_TRUE_TEST,True_ozone),dim='longitude')
        R2_TEST  = xr.concat((R2_TEST,R2_test),dim='longitude')

z_pre = OZONE_PRED_TEST.mean(dim=['t','longitude'],skipna=True).squeeze()
z = OZONE_TRUE_TEST.mean(dim=['t','longitude'],skipna=True).squeeze()
z_pre["hybrid_ht"] = z_pre["hybrid_ht"]/1000
z["hybrid_ht"] = z["hybrid_ht"]/1000
r2_test = R2_TEST.mean(dim=['longitude'],skipna=True).squeeze()
r2_test["hybrid_ht"] = r2_test["hybrid_ht"]/1000


#%% plot 10yrs mean Predicted & True Ozone
vmin=0; vmax=11
fig, axs = plt.subplots(1, 2, figsize=(16, 8))
plt.rcParams.update({'font.size': 18})
cmap = "jet"
cf1 = xr.plot.contourf(z_pre,ax=axs[0],vmin=vmin,vmax=vmax,levels=41,cmap=cmap,add_colorbar=False)
axs[0].set_title("10yrs mean O3 from Ridge Reg (single column input)")
axs[0].set_ylabel("Height(km)")
axs[0].set_xticks(ticks=list(np.arange(-90,90,30))+[86.875])
axs[0].set_xticklabels(["90S","60S","30S","Eq","30N","60N","90N"])

cf2 = xr.plot.contourf(z,ax=axs[1],vmin=vmin,vmax=vmax,levels=41,cmap=cmap,add_colorbar=False)
axs[1].set_title("From UKESM")
axs[1].set_ylabel("")
axs[1].set_xticks(ticks=list(np.arange(-90,90,30))+[86.875])
axs[1].set_xticklabels(["90S","60S","30S","Eq","30N","60N","90N"])

fig.colorbar(cf2, ax=axs, location="bottom",use_gridspec=True,fraction=0.05, aspect=20*2,
                extend="both", orientation='horizontal', pad=0.11)
plt.savefig(Outfilepath+"Ozone_Ridge_Reg_with_zonal_input.png", dpi=400)
plt.close()

### plot R2 score on the lat-lev cross section
vmin, vmax=0, 0.9
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
plt.rcParams.update({'font.size': 18})# must set in top
cf1 = xr.plot.contourf(r2_test,ax=ax,vmin=vmin,vmax=vmax,levels=41,cmap=cmap,add_colorbar=False)
ax.set_title("Ridge R2 (single column)")
ax.set_ylabel("Height(km)")
ax.set_xticks(ticks=list(np.arange(-90,90,30))+[86.875])
ax.set_xticklabels(["90S","60S","30S","Eq","30N","60N","90N"])
# cax = fig.add_axes([0.9, 0.11, 0.05, 0.76])
fig.colorbar(mappable=cf1,ax=ax,orientation='vertical')
plt.savefig(Outfilepath+"R2_Ridge_Reg_with_zonal_input.png", dpi=400)
plt.close()

# %%
### print out Predicted ozone on test set to ncfile
OZONE_PRED_TEST=OZONE_PRED_TEST.transpose("t","hybrid_ht","latitude","longitude")
OZONE_PRED_TEST.to_netcdf(Outfilepath+"/Offline_predicted_O3_with_zonal_input_2000s.nc","w")

R2_TEST = R2_TEST.transpose("hybrid_ht","latitude","longitude")
R2_TEST.to_netcdf(Outfilepath+"/Offline_R2_score_with_zonal_input_2000s.nc","w")

