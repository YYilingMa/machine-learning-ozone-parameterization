###
# Predict ozone with Feedforward Neural Network
# Used for Figure S1 in the supporting information

#%% 
import numpy as np
import xarray as xr
import os
from scipy.stats import spearmanr
import pandas as pd
import cftime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
from torch.utils.data import DataLoader, TensorDataset
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import shutil
from tqdm import tqdm
import glob
from torchsummary import summary

add_feature1 = "hum" # ADD which fearture: hum, p, LW, SW or None
add_feature2 = None # ADD which fearture: hum, p, LW, SW or None
add_feature3 = None
nt_train = 360*32 # training set length
nt_val = 360*8
nt_test = 360*10

epoch_size = 40 # 40
lat_int = 3 # 3
lon_sel = [0.9375, 57.1875, 113.4375, 169.6875, 225.9375, 282.1875, 338.4375]
lr = 0.00005
dropout_prob = 0.2 # During training, randomly zeroes some of the elements of the input tensor with probability p
n_hidden_layers = 1
hidden_layer_size = 20
Outfilepath = "offline_NN_ZonalR2_add_q_4/"
Infilepath = "/hkfs/work/workspace/scratch/ou4895-ukesm_restored/ou4895-ukesm2-1761661744/ou4895-ukesm2-1755999483/data/"


# Determine if there's a GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} for NN".format(device))

if os.path.exists(Outfilepath):
    state_filedirs = sorted(list(glob.glob(Outfilepath+'NNmodel*')))
    for j in state_filedirs:
        shutil.rmtree(j)
    print("The NNmodel*/ in directory ./{} is cleared!".format(Outfilepath))
else:
    os.makedirs(Outfilepath)
    print("The new directory {} is created!".format(Outfilepath))

seed = 11 # Used to fix the random seed so that the same results can be obtained for each independent training
def seed_worker(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False # False:cuDNN deterministically select an algorithm
    # possibly at the cost of reduced performance
    torch.use_deterministic_algorithms(False) #

def R2_score(x_pred,x_true):
    # Best possible score is 1.0, lower values are worse.
    # https://scikit-learn.org/0.15/modules/generated/sklearn.linear_model.RidgeCV.html
    u = ((x_pred-x_true)**2).sum(axis=0)
    v = ((x_true-x_true.mean(axis=0))**2).sum(axis=0)
    return 1-u/v

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class FeedForwardNN(nn.Module):
    def __init__(self, input_shape, num_hidden_layers, hidden_layer_size, dropout_prob):
        super(FeedForwardNN, self).__init__()
        self.flatten = nn.Flatten()
        self.input_shape = input_shape
        self.input_size = input_shape[0] * input_shape[1] #* input_shape[2]

        # Define the layers explicitly
        self.input_layer = nn.Linear(self.input_size, hidden_layer_size)
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_layer_size, hidden_layer_size),
                nn.Dropout(dropout_prob)  # Add dropout after each hidden layer
            ) for _ in range(num_hidden_layers - 1)
        ])
        self.output_layer = nn.Linear(hidden_layer_size, 1) # output size=1

    def forward(self, x):
        x = x.float()
        x = self.flatten(x)  # Flatten the input using reshape
        x = F.elu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.elu(layer[0](x)) # Apply ELU activation function to the linear layer
            x = layer[1](x) # Apply dropout after each hidden layer
        x = self.output_layer(x)  # Output without activation function
        #x = F.relu(x)
        x = x.reshape(-1, 1)  # Reshape back to the original input shape using reshape
        return x

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=8, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train_model(id_lon,id_lat,id_h, epoch, train_loader, val_loader, device, model, criterion, optimizer):

    model = model.to(device)

    # initialize the early_stopping object
    model_path = Outfilepath+r'NNmodel_Lon'+str(id_lon)+'_Lat'+str(id_lat)+'_H'+str(id_h)+'/'   
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    file = f"NN.pth"
    file_path = os.path.join(model_path, file)
    early_stopping = EarlyStopping(patience=8, verbose=False, path=file_path)

    mae = nn.L1Loss() # mean absolute error
    for e in range(epoch):

        model.train() 
        train_loss = 0.0
        train_mae = 0.0
        train_batches = 0
        # train_loader = tqdm(train_loader)  
        # train_loader.set_description(f'[Train Epoch:{e+1:04d}/{epoch:04d} lr:{get_lr(optimizer):.6f}]')
        for (i, (x,y)) in enumerate(train_loader,0):  
            inputs, labels = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)#.squeeze()
            # loss
            loss = criterion(outputs, labels)
            # Compute gradients
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_mae += mae(outputs, labels).item()
            train_batches += 1
            # postfix = {
            #     'train_loss': f'{train_loss / (i + 1):.6f}',
            #     'train_mae': f'{train_mae / (i + 1):.6f}',
            #     }
            # train_loader.set_postfix(log=postfix)
        
        train_loss_epoch = train_loss / train_batches
        train_mae_epoch = train_mae / train_batches

        # ---------- Validation ----------
        model.eval()  # convert to validation mode
        val_loss = 0.0
        val_mae = 0.0
        val_batches = 0

        with torch.no_grad(): 
            # val_loader = tqdm(val_loader) 
            # val_loader.set_description(f'[val Epoch:{e+1:04d}/{epoch:04d}]')
            for (i, (x,y)) in enumerate(val_loader, 0):
                inputs, labels = x.to(device), y.to(device)
                outputs = model(inputs)#.squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_mae += mae(outputs, labels).item()
                val_batches += 1
                # postfix = {
                #     'val_loss': f'{val_loss / (i + 1):.6f}',
                #     'val_mae': f'{val_mae / (i + 1):.6f}',
                #     }
                # val_loader.set_postfix(log=postfix)
        val_loss_epoch = val_loss / val_batches
        val_mae_epoch = val_mae / val_batches

        # ---------- Print epoch summary ----------
        print(f"Epoch [{e+1}/{epoch}] "
              f"Train Loss: {train_loss_epoch:.6f}, Train MAE: {train_mae_epoch:.6f} | "
              f"Val Loss: {val_loss_epoch:.6f}, Val MAE: {val_mae_epoch:.6f}")
        
        early_stopping(val_loss_epoch, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print('Finished Training!')

def evaluate_model(test_loader, device, model):

    model.eval()  # sets the PyTorch model to evaluation mode, disabling operations like dropout, useful for inference and testing.
    predictions = []
    true_values = []
    with torch.no_grad():
        for (x, y) in tqdm(test_loader, desc="Testing"):
            inputs, labels = x.to(device), y.to(device)
            outputs = model(inputs).squeeze(-1)
            predictions.append(outputs.cpu().numpy())
            true_values.append(labels.cpu().numpy())

    predictions1 = np.concatenate(predictions, axis=0)
    true_values1 = np.concatenate(true_values, axis=0)

    return predictions1, true_values1

#%% load data
### Ozone
ozone_file = xr.open_dataset(Infilepath+'fullChem_ozone_piCTRL_1962_2011_UKESM.nc',
                                  decode_times=True).sel(longitude=lon_sel, method="nearest")
Ozone_train = ozone_file['field2101'][1:nt_train,:76,::lat_int,:]*1e6/1.657
Ozone_val = ozone_file['field2101'][nt_train+1:nt_train+nt_val,:76,::lat_int,:]*1e6/1.657
Ozone_test = ozone_file['field2101'][1-nt_test:,:76,::lat_int,:]*1e6/1.657
del ozone_file
print("end of loading ozone data")

### Temp
temp_file = xr.open_dataset(Infilepath+'fullChem_temp_piCTRL_1962_2011_UKESM.nc',
                            decode_times=False).sel(longitude=lon_sel, method="nearest")
Temp_train=temp_file['temp'][:nt_train-1,:76,::lat_int,:]
Temp_val=temp_file['temp'][nt_train:nt_train+nt_val-1,:76,::lat_int,:]
Temp_test=temp_file['temp'][-nt_test:-1,:76,::lat_int,:]
del temp_file
print("end of loading temp data")

oz_train_ls = []; oz_val_ls=[]; oz_test_ls = []
tp_train_ls = []; tp_val_ls = []; tp_test_ls = []
for i in range(len(lon_sel)):
    tp_train_ls.append(Temp_train.sel(longitude=lon_sel[i], method="nearest"))
    tp_val_ls.append(Temp_val.sel(longitude=lon_sel[i], method="nearest"))
    tp_test_ls.append(Temp_test.sel(longitude=lon_sel[i], method="nearest"))
    oz_train_ls.append(Ozone_train.sel(longitude=lon_sel[i], method="nearest"))
    oz_val_ls.append(Ozone_val.sel(longitude=lon_sel[i], method="nearest"))
    oz_test_ls.append(Ozone_test.sel(longitude=lon_sel[i], method="nearest"))

### Add_feature data
ih_sel_tem = np.concatenate((np.arange(10,36,4), np.arange(36,75,1), np.arange(75,80,2)))
if add_feature1 != None:
    add_file1 = xr.open_dataset(Infilepath+'/fullChem_'+add_feature1+'_piCTRL_1962_2011_UKESM.nc',decode_times=False).sel(longitude=lon_sel,method="nearest")
    if add_feature1=="hum": # ADD which fearture: hum, p, LW, SW
        valname = 'q'
    elif add_feature1=="p":
        valname = 'p'
    elif add_feature1=="LW":
        valname = 'lwhr'    
    elif add_feature1=="SW":
        valname = 'swhr'
    add_train1 = add_file1[valname][:nt_train-1,ih_sel_tem,::lat_int,:]
    add_val1 = add_file1[valname][nt_train:nt_train+nt_val-1,ih_sel_tem,::lat_int,:]
    add_test1 = add_file1[valname][-nt_test:-1,ih_sel_tem,::lat_int,:]

    add_train1_ls = []; add_val1_ls = []; add_test1_ls = []
    for i in range(len(lon_sel)):
        add_train1_ls.append(add_train1.sel(longitude=lon_sel[i], method="nearest"))
        add_val1_ls.append(add_val1.sel(longitude=lon_sel[i], method="nearest"))
        add_test1_ls.append(add_test1.sel(longitude=lon_sel[i], method="nearest"))  

    if add_feature2 != None:
        add_file2 = xr.open_dataset(Infilepath+'/fullChem_'+add_feature2+'_piCTRL_1962_2011_UKESM.nc',decode_times=False).sel(longitude=lon_sel,method="nearest")
        if add_feature2=="hum": # ADD which fearture: hum, p, LW, SW
            valname = 'q'
        elif add_feature2=="p":
            valname = 'p'
        elif add_feature2=="LW":
            valname = 'lwhr'    
        elif add_feature2=="SW":
            valname = 'swhr'
        add_train2 = add_file2[valname][:nt_train-1,ih_sel_tem,::lat_int,:]
        add_val2 = add_file2[valname][nt_train:nt_train+nt_val-1,ih_sel_tem,::lat_int,:]
        add_test2 = add_file2[valname][-nt_test:-1,ih_sel_tem,::lat_int,:]

        add_train2_ls = []; add_val2_ls = []; add_test2_ls = []
        for i in range(len(lon_sel)):
            add_train2_ls.append(add_train2.sel(longitude=lon_sel[i], method="nearest"))
            add_val2_ls.append(add_val2.sel(longitude=lon_sel[i], method="nearest"))  
            add_test2_ls.append(add_test2.sel(longitude=lon_sel[i], method="nearest"))  

        if add_feature3 != None:
            add_file3 = xr.open_dataset('./fullChem_'+add_feature3+'_piCTRL_1962_2011_UKESM.nc',decode_times=False).sel(longitude=lon_sel,method="nearest")
            if add_feature3=="hum": # ADD which fearture: hum, p, LW, SW
                valname = 'q'
            elif add_feature3=="p":
                valname = 'p'
            elif add_feature3=="LW":
                valname = 'lwhr'    
            elif add_feature3=="SW":
                valname = 'swhr'
            add_train3 = add_file3[valname][:nt_train-1,ih_sel_tem,::lat_int,:]
            add_val3 = add_file3[valname][nt_train:nt_train+nt_val-1,ih_sel_tem,::lat_int,:]
            add_test3 = add_file3[valname][-nt_test:-1,ih_sel_tem,::lat_int,:]

            add_train3_ls = []; add_val3_ls = []; add_test3_ls = []
            for i in range(len(lon_sel)):
                add_train3_ls.append(add_train3.sel(longitude=lon_sel[i], method="nearest"))
                add_val3_ls.append(add_val3.sel(longitude=lon_sel[i], method="nearest"))  
                add_test3_ls.append(add_test3.sel(longitude=lon_sel[i], method="nearest"))  


#%% train and test
criterion = nn.L1Loss()
for i in range(len(lon_sel)):

    X_train = np.array(tp_train_ls[i])
    X_val = np.array(tp_val_ls[i])
    X_test = np.array(tp_test_ls[i])
    Y_train = np.array(oz_train_ls[i])
    Y_val = np.array(oz_val_ls[i])
    Y_test = np.array(oz_test_ls[i])

    if add_feature1 != None:
        X_train = np.concatenate((X_train,np.array(add_train1_ls[i])), axis=1)
        X_val = np.concatenate((X_val,np.array(add_val1_ls[i])), axis=1)
        X_test = np.concatenate((X_test,np.array(add_test1_ls[i])), axis=1)
        if add_feature2 != None:
            X_train = np.concatenate((X_train,np.array(add_train2_ls[i])), axis=1)
            X_val = np.concatenate((X_val,np.array(add_val2_ls[i])), axis=1)
            X_test = np.concatenate((X_test,np.array(add_test2_ls[i])), axis=1)            
            if add_feature3 != None:
                X_train = np.concatenate((X_train,np.array(add_train3_ls[i])), axis=1)
                X_val = np.concatenate((X_val,np.array(add_val3_ls[i])), axis=1)
                X_test = np.concatenate((X_test,np.array(add_test3_ls[i])), axis=1)    

    _, nx_feature, n_lat = X_train.shape
    _, n_h, _ = Y_train.shape

    ozone_pred_test = np.empty((nt_test-1, n_h, n_lat))
    R2_test_section = np.empty((n_h, n_lat))
    for j in range(n_lat):
        for k in range(n_h):

            X_train_sel = X_train[:,:,j].reshape((nt_train-1,-1))
            X_val_sel = X_val[:,:,j].reshape((nt_val-1,-1))
            X_test_sel = X_test[:,:,j].reshape((nt_test-1,-1))
            Y_train_sel = Y_train[:,k,j].reshape((nt_train-1,-1))
            Y_val_sel = Y_val[:,k,j].reshape((nt_val-1,-1))
            Y_test_sel = Y_test[:,k,j].reshape((nt_test-1,-1))

            ### Standardize
            scaler_x=StandardScaler()
            scaler_x.fit(X_train_sel)
            X_train_std = scaler_x.transform(X_train_sel)
            X_val_std = scaler_x.transform(X_val_sel)
            X_test_std = scaler_x.transform(X_test_sel)

            scaler_y=StandardScaler()
            scaler_y.fit(Y_train_sel)
            Y_train_std = scaler_y.transform(Y_train_sel)
            Y_val_std = scaler_y.transform(Y_val_sel)
            Y_test_std = scaler_y.transform(Y_test_sel)

            X_train_ts = torch.tensor(X_train_std, dtype=torch.float32).to(device)# (time, height)
            X_val_ts = torch.tensor(X_val_std, dtype=torch.float32).to(device)
            X_test_ts = torch.tensor(X_test_std, dtype=torch.float32).to(device)
            Y_train_ts = torch.tensor(Y_train_std, dtype=torch.float32).to(device)
            Y_val_ts = torch.tensor(Y_val_std, dtype=torch.float32).to(device)
            Y_test_ts = torch.tensor(Y_test_std, dtype=torch.float32).to(device)

            train_dataset = TensorDataset(X_train_ts, Y_train_ts)
            val_dataset = TensorDataset(X_val_ts, Y_val_ts)
            test_dataset = TensorDataset(X_test_ts, Y_test_ts)

            print(f'train num: {train_dataset.__len__()}, val num: {val_dataset.__len__()}, test num: {test_dataset.__len__()}')

            g = torch.Generator()
            g.manual_seed(seed)
            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False,worker_init_fn=seed_worker(seed),generator=g)
            val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False,worker_init_fn=seed_worker(seed),generator=g)
            test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False,worker_init_fn=seed_worker(seed),generator=g)

            model = FeedForwardNN(input_shape=(1,nx_feature), num_hidden_layers=n_hidden_layers, hidden_layer_size=hidden_layer_size,dropout_prob=dropout_prob).to(device)
            summary(model, input_size=(1, nx_feature))
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            # Training
            train_model(i, j, k, epoch_size, train_loader, val_loader, device, model, criterion, optimizer)
            # Testing
            ozone_pred_test_std, ozone_true_test_std = evaluate_model(test_loader, device, model) # standardized values
            ozone_pred_test_std=ozone_pred_test_std.reshape(-1,1)
            # R2 score    
            R2_test_section[k,j] = R2_score(ozone_pred_test_std,ozone_true_test_std)
            # inverse normalization
            ozone_pred_test[:,k,j] = scaler_y.inverse_transform(ozone_pred_test_std).reshape((-1))

    True_ozone = oz_test_ls[i]
    lon = True_ozone["longitude"].values

    Pred_ozone = xr.DataArray(
    ozone_pred_test,
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
        OZONE_PRED_TEST = Pred_ozone
        OZONE_TRUE_TEST = True_ozone
        R2_TEST = R2_test
    else:
        OZONE_PRED_TEST = xr.concat((OZONE_PRED_TEST,Pred_ozone),dim='longitude')
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
plt.rcParams.update({'font.size': 18})# must set in top
cmap = "jet"
cf1 = xr.plot.contourf(z_pre,ax=axs[0],vmin=vmin,vmax=vmax,levels=41,cmap=cmap,add_colorbar=False)
axs[0].set_title("10yrs mean O3 from NN (single column input)")
axs[0].set_ylabel("Height(km)")
axs[0].set_xticks(ticks=list(np.arange(-90,90,30))+[86.875])
axs[0].set_xticklabels(["90S","60S","30S","Eq","30N","60N","90N"])

cf2 = xr.plot.contourf(z,ax=axs[1],vmin=vmin,vmax=vmax,levels=41,cmap=cmap,add_colorbar=False)
axs[1].set_title("Full chem")
axs[1].set_ylabel("")
axs[1].set_xticks(ticks=list(np.arange(-90,90,30))+[86.875])
axs[1].set_xticklabels(["90S","60S","30S","Eq","30N","60N","90N"])

fig.colorbar(cf2, ax=axs, location="bottom",use_gridspec=True,fraction=0.05, aspect=20*2,
                extend="both", orientation='horizontal', pad=0.11)
plt.savefig(Outfilepath+"Ozone_NN_with_single_column_input.png", dpi=400)

### plot R2 score on the lat-lev cross section
vmin, vmax=0, 0.9
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
plt.rcParams.update({'font.size': 18})# must set in top
cf1 = xr.plot.contourf(r2_test,ax=ax,vmin=vmin,vmax=vmax,levels=41,cmap=cmap,add_colorbar=False)
ax.set_title("Ridge R2(single column)")
ax.set_ylabel("Height(km)")
ax.set_xticks(ticks=list(np.arange(-90,90,30))+[86.875])
ax.set_xticklabels(["90S","60S","30S","Eq","30N","60N","90N"])
fig.colorbar(mappable=cf1,ax=ax,orientation='vertical')
plt.savefig(Outfilepath+"R2_NN_with_single_column_input.png", dpi=400)

# %%
### print out predicted ozone on test set to ncfile
OZONE_PRED_TEST=OZONE_PRED_TEST.transpose("t","hybrid_ht","latitude","longitude")
OZONE_PRED_TEST.to_netcdf(Outfilepath+"/Offline_NN_predicted_O3_with_single_column_input_2000s.nc","w")

R2_TEST = R2_TEST.transpose("hybrid_ht","latitude","longitude")
R2_TEST.to_netcdf(Outfilepath+"/Offline_NN_R2_score_with_single_column_input_2000s.nc","w")


# %%
