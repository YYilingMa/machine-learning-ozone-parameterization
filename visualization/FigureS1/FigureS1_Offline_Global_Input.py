# use global temp to predict ozone on selected longitudes
# Used for Figure S1 in the supporting information

#%%
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.setrecursionlimit(1000000) # Otherwise, joblib will not save deep nets.                                                                                             
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.decomposition import IncrementalPCA
import xarray as xr
import faulthandler; faulthandler.enable() # faulthandler will help to debug you 
# or will try to show you nearest line in your code which caused the segmentation fault.
np.set_printoptions(precision=4, suppress=True)
import multiprocessing as mp


l_load_X = False # load X_train and X_test data
l_load_pca = False # True: load X_train_pca & X_test_pca; False: save X_train_pca & X_test_pca
add_feature1 = None # ADD which fearture: hum, p, LW, SW or None
add_feature2 = None # ADD which fearture: hum, p, LW, SW or None
nt_train = 360*40 # training set length
nt_test = 360*10
lat_int = 3
lon_int = 30 # chosen lons: 0.9375,  57.1875, 113.4375, 169.6875, 225.9375, 282.1875, 338.4375
Outfilepath = "output_global_R2_coarser/"
Infilepath = "/hkfs/work/workspace/scratch/ou4895-ukesm_restored/ou4895-ukesm2-1761661744/ou4895-ukesm2-1755999483/data/"

if not os.path.exists(Outfilepath):
    os.makedirs(Outfilepath)
    print("The new directory ./{} is created!".format(Outfilepath))

def R2_score(x_pred,x_true):
    u = ((x_pred-x_true)**2).sum(axis=0)
    v = ((x_true-x_true.mean(axis=0))**2).sum(axis=0)
    return 1-u/v

def test_std(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)
    return X_train_std, X_test_std

def pca_on_X(X_train, X_test):

    n_component = 3000 # exp ratio = 0.986. 3800 components of pca, 2310
    batch_size = 3780

    print("X_train.shape: ", X_train.shape)

    n_component = min(n_component, X_train.shape[1])
    ipca = IncrementalPCA(n_components=n_component, copy=False)
    length = X_train.shape[0]
    X_train_pca = np.zeros(shape=(length, ipca.n_components))
    for i in range(0, length, batch_size):
        print("start to fit on batch {} - {}".format(i,i+batch_size))
        X_batch = X_train[i:i+batch_size].copy()
        ipca.partial_fit(X_batch)

    X_train_pca = ipca.transform(X_train)
    print("X_train_pca.shape: ", X_train_pca.shape)
    print('pca.explained_variance_ratio_',np.sum(ipca.explained_variance_ratio_))
    X_test_pca = ipca.transform(X_test)

    return X_train_pca, X_test_pca
    
if add_feature1 != None:
    if add_feature2 != None:
        print("Using Temp+{}+{} as input features!".format(add_feature1,add_feature2))
    else:
        print("Using Temp+{} as input features!".format(add_feature1))
else:
    print("Using only Temp as input feature!")

#%% Temp data preparation
if (add_feature1 == None) and (add_feature2 == None):
    train_file = "X_train_global_pca"
    test_file = "X_test_global_pca"
elif (add_feature1 != None) and (add_feature2 == None):
    train_file = "X_train_global_pca(+{})".format(add_feature1)
    test_file = "X_test_global_pca(+{})".format(add_feature1)
elif (add_feature1 != None) and (add_feature2 != None):
    train_file = "X_train_global_pca(+{}+{})".format(add_feature1,add_feature2)
    test_file = "X_test_global_pca(+{}+{})".format(add_feature1,add_feature2)

if l_load_pca:
    X_train_pca = np.load(Infilepath+train_file+".npy")
    X_test_pca = np.load(Infilepath+test_file+".npy")
else:
    print("Reading temp data now")
    if l_load_X:
        X_train=np.load(Infilepath+"X_train_temp.npy")
        X_test=np.load(Infilepath+"X_test_temp.npy")
    else:
        temp_file = xr.open_dataset(Infilepath+'fullChem_temp_piCTRL_1962_2034_UKESM.nc',decode_times=False)
        ih_sel_tem = np.concatenate((np.arange(10,36,4),np.arange(36,75,1),np.arange(75,80,2)))
        Temp_train = temp_file['temp'][:nt_train-1,ih_sel_tem,::lat_int,::lon_int]
        Temp_test = temp_file['temp'][nt_train:nt_train+nt_test-1,ih_sel_tem,::lat_int,::lon_int]
        X_train = np.array(Temp_train).reshape((nt_train-1,-1))
        X_test = np.array(Temp_test).reshape((nt_test-1,-1))
        del temp_file, Temp_train, Temp_test
        np.save(Infilepath+"X_train_temp.npy", X_train)
        np.save(Infilepath+"X_test_temp.npy", X_test)

    if add_feature1 != None:
        print("Reading {} data now".format(add_feature1))
        if l_load_X:
            X_train1=np.load("X_train_{}.npy".format(add_feature1))
            X_test1=np.load("X_test_{}.npy".format(add_feature1))
        else:
            add_file1 = xr.open_dataset(Infilepath+'/fullChem_'+add_feature1+'_piCTRL_1962_2034_UKESM.nc',decode_times=False)
            if add_feature1=="hum": # ADD which fearture: hum, p, LW, SW
                valname = 'q'
            elif add_feature1=="p":
                valname = 'p'
            elif add_feature1=="LW":
                valname = 'lwhr'    
            elif add_feature1=="SW":
                valname = 'swhr'
            add_train1 = add_file1[valname][:nt_train-1,ih_sel_tem,::lat_int,::lon_int]
            add_test1 = add_file1[valname][nt_train:nt_train+nt_test-1,ih_sel_tem,::lat_int,::lon_int]
            X_train1 = np.array(add_train1).reshape((nt_train-1,-1))
            X_test1 = np.array(add_test1).reshape((nt_test-1,-1))
            np.save(Infilepath+"X_train_{}.npy".format(add_feature1),X_train1)
            np.save(Infilepath+"X_test_{}.npy".format(add_feature1),X_test1)
            del add_file1, add_train1, add_test1
        X_train = np.concatenate((X_train,X_train1), axis=1)
        X_test = np.concatenate((X_test,X_test1), axis=1)
        del X_train1, X_test1

        if add_feature2 != None:
            print("Reading {} data now".format(add_feature2))
            if l_load_X:
                X_train2=np.load(Infilepath+"X_train_{}.npy".format(add_feature2))
                X_test2=np.load(Infilepath+"X_test_{}.npy".format(add_feature2))
            else:
                add_file2 = xr.open_dataset(Infilepath+'/fullChem_'+add_feature2+'_piCTRL_1962_2034_UKESM.nc',decode_times=False)
                if add_feature2=="hum": # ADD which fearture: hum, p, LW, SW
                    valname = 'q'
                elif add_feature2=="p":
                    valname = 'p'
                elif add_feature2=="LW":
                    valname = 'lwhr'    
                elif add_feature2=="SW":
                    valname = 'swhr'
                add_train2 = add_file2[valname][:nt_train-1,ih_sel_tem,::lat_int,::lon_int]
                add_test2 = add_file2[valname][nt_train:nt_train+nt_test-1,ih_sel_tem,::lat_int,::lon_int]
                X_train2 = np.array(add_train2).reshape((nt_train-1,-1))
                X_test2 = np.array(add_test2).reshape((nt_test-1,-1))
                del add_file2, add_train2, add_test2
            X_train = np.concatenate((X_train,X_train2), axis=1)
            X_test = np.concatenate((X_test,X_test2), axis=1)    
            del X_train2, X_test2
                
    ### Standardize
    scaler=StandardScaler()
    scaler.fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)
    del X_train, X_test

    ### dimensionality reduction using PCA
    X_train_pca, X_test_pca = pca_on_X(X_train_std, X_test_std)
    del X_train_std, X_test_std # clear memory
    np.save(Infilepath+train_file+".npy", X_train_pca)
    np.save(Infilepath+test_file+".npy", X_test_pca)

#%%
### Ozone data preparation
ozone_file = xr.open_dataset(Infilepath+'fullChem_ozone_piCTRL_1962_2034_UKESM.nc',decode_times=True)
ih_sel_o3 = np.concatenate((np.arange(10,36,6),np.arange(36,76,2),np.arange(76,80,4)))
Ozone_train = ozone_file['field2101'][1:nt_train,ih_sel_o3,::lat_int,::lon_int]*1e6/1.657 # MASS MIXING RATIO kg/kg-1 -> ppmv
Ozone_test = ozone_file['field2101'][1+nt_train:nt_train+nt_test,ih_sel_o3,::lat_int,::lon_int]*1e6/1.657 # MASS MIXING RATIO kg/kg-1 -> ppmv

#%% Training and Testing
print("Using Global data as input")
alpha_i = [1000,10000,20000,40000,60000,80000,100000,120000,160000,240000,320000,640000,1280000]
parameters = {
    'alpha': alpha_i,
    'fit_intercept': [True],
    'max_iter':[1000],
    'random_state':[100] #If int, random_state is the seed used by the random number generator.
     # If None, the random number generator is the RandomState instance used by np.random. 
             }
cv_obj = KFold(n_splits=5) # Each fold is then used once as a validation while the k - 1 remaining folds form the training set.

Y_train = np.array(Ozone_train).reshape((nt_train-1,-1))
Y_test = np.array(Ozone_test).reshape((nt_test-1,-1))
n_t_test, n_h, n_lat, n_lon = Ozone_test.shape

### Standardize
scaler_y=StandardScaler()
scaler_y.fit(Y_train)
Y_train_std = scaler_y.transform(Y_train)
Y_test_std = scaler_y.transform(Y_test)
del Y_train, Y_test # clear memory

### GridSearch
regr_ijk = GridSearchCV(Ridge(),parameters,cv=cv_obj,n_jobs=-1,refit=True)
# refit : bool, str, or callable, default=True; Refit an estimator using the best found parameters on the whole dataset.
# n_jobs: Number of jobs to run in parallel.  If set to -1, all CPUs are used
# refit:  Refit an estimator using the best found parameters on the whole dataset.
# cv: Determines the cross-validation splitting strategy.

n_dim = n_h*n_lat*n_lon
best_alpha_array = []
best_coef_array = []
y_pred_train_array = []
y_pred_test_array = []
y_true_train_array = []
y_true_test_array = []

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
  
R2_test_section = R2_score(ozone_pred_test,ozone_true_test).reshape((n_h,n_lat,n_lon))    

# inversed normalization
predicted_ozone_r = scaler_y.inverse_transform(ozone_pred_test)
Predicted_ozone = predicted_ozone_r.reshape(n_t_test,n_h,n_lat,n_lon)
True_ozone = Ozone_test

#%%
Ozone_pred = xr.DataArray(
Predicted_ozone,
dims = True_ozone.dims,
coords = True_ozone.coords,
name="Ozone_predicted_by_Ridge"
)

R2_test = xr.DataArray(
R2_test_section,
dims = ['hybrid_ht', 'latitude', 'longitude'],
coords = True_ozone[-1].coords,
name="R2_test"
) 

z_pre = Ozone_pred.mean(dim=['t','longitude'],skipna=True)
z = True_ozone.mean(dim=['t','longitude'],skipna=True)
R2_zonal = R2_test.mean(dim=['longitude'],skipna=True)
z_pre["hybrid_ht"] = z_pre["hybrid_ht"]/1000
z["hybrid_ht"] = z["hybrid_ht"]/1000
R2_zonal["hybrid_ht"] = R2_zonal["hybrid_ht"]/1000

### plot Predicted & True Ozone
vmin=0; vmax=11
plt.rcParams.update({'font.size': 18})
fig, axs = plt.subplots(1, 2, figsize=(16, 8))
cmap = "jet" 
cf1 = xr.plot.contourf(z_pre,ax=axs[0],vmin=vmin,vmax=vmax,levels=41,cmap=cmap,add_colorbar=False)
axs[0].set_title("10yrs mean ozone from Ridge Reg (global input)")
axs[0].set_ylabel("Height(km)")
axs[0].set_xticks(ticks=list(np.arange(-90,90,30))+[88.125])
axs[0].set_xticklabels(["90S","60S","30S","Eq","30N","60N","90N"])

cf2 = xr.plot.contourf(z,ax=axs[1],vmin=vmin,vmax=vmax,levels=41,cmap=cmap,add_colorbar=False)
axs[1].set_title("From UKESM")
axs[1].set_ylabel("")
axs[1].set_xticks(ticks=list(np.arange(-90,90,30))+[88.125])
axs[1].set_xticklabels(["90S","60S","30S","Eq","30N","60N","90N"])
fig.colorbar(cf2, ax=axs, location="bottom",use_gridspec=True,fraction=0.05, aspect=20*2,
                extend="both", orientation='horizontal', pad=0.11)
plt.savefig(Outfilepath+"Ozone_Ridge_Reg_with_global_input.png", dpi=400)
plt.show()

### plot R2 score on the lat-lev cross section
vmin, vmax=0, 0.9
plt.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
cf1 = xr.plot.contourf(R2_zonal,ax=ax,vmin=vmin,vmax=vmax,levels=41,cmap=cmap,add_colorbar=False)
ax.set_title("Ridge R2 (global)")
ax.set_ylabel("Height(km)")
ax.set_xticks(ticks=list(np.arange(-90,90,30))+[88.125])
ax.set_xticklabels(["90S","60S","30S","Eq","30N","60N","90N"])
fig.colorbar(mappable=cf1,ax=ax,orientation='vertical')
plt.savefig(Outfilepath+"R2_Ridge_Reg_on_with_global_input.png", dpi=400)
plt.show()

# %%
# print out Predicted ozone on test set to ncfile
OZONE_PRED_TEST=Ozone_pred.transpose("t","hybrid_ht","latitude","longitude")
OZONE_PRED_TEST.to_netcdf(Outfilepath+"/Offline_predicted_O3_with_global_input_2000s.nc","w")

R2_TEST = R2_test.transpose("hybrid_ht","latitude","longitude")
R2_TEST.to_netcdf(Outfilepath+"/Offline_R2_score_with_global_input_2000s.nc","w")

# %%
