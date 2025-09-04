###
# predict ozone with online temp
# test on 2000
# temp and ozone are both standardized
# zonal mean

#%% 
import numpy as np
import xarray as xr
import os
from scipy.stats import spearmanr
import pandas as pd
import cftime
# import netCDF4
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
import os
from torchsummary import summary

Outfilepath = "offline_NN/"
nt_train = 360*32 # training set length
nt_val = 360*8
nt_test = 360*10
lr = 0.00005
dropout_prob = 0.2 # During training, randomly zeroes some of the elements of the input tensor with probability p
n_hidden_layers = 1
hidden_layer_size = 40


# Determine if there's a GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} for NN".format(device))

if os.path.exists(Outfilepath):
    # os.remove(output_dir+'Unetmodel*')
    state_filedirs = sorted(list(glob.glob(Outfilepath+'NNmodel*')))
    for j in state_filedirs:
        shutil.rmtree(j)
    # os.makedirs(output_dir)
    print("The NNmodel*/ in directory ./{} is cleared!".format(Outfilepath))
else:
    os.makedirs(Outfilepath)
    print("The new directory {} is created!".format(Outfilepath))

seed = 11 # Used to fix the random seed so that the same results can be obtained for each independent training, old:11
def seed_worker(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False #False:cuDNN deterministically select an algorithm
    # possibly at the cost of reduced performance
    torch.use_deterministic_algorithms(False) #


#%% 
################################################################
###                      Model Definition                    ###
################################################################   
### Standadized input&ouput
def Standard(data_train,data_val,data_test):
    data_train_np = np.array(data_train)
    size_train = data_train_np.shape
    data_train_r = data_train_np.reshape(size_train[0],-1)

    data_val_np = np.array(data_val)
    size_val = data_val_np.shape
    data_val_r = data_val_np.reshape(size_val[0],-1)

    data_test_np = np.array(data_test)
    size_test = data_test_np.shape
    data_test_r = data_test_np.reshape(size_test[0],-1)

    scaler=StandardScaler()
    scaler.fit(data_train_r)
    data_train_std = scaler.transform(data_train_r)
    data_val_std = scaler.transform(data_val_r)
    data_test_std = scaler.transform(data_test_r)

    return data_train_std.reshape(size_train),data_val_std.reshape(size_val),data_test_std.reshape(size_test),scaler


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
        x = x.reshape(-1, self.input_size)  # Flatten the input using reshape
        x = F.elu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.elu(layer[0](x)) # Appply ELU activation function to the linear layer
            x = layer[1](x) # Apply dropout after each hidden layer
        x = self.output_layer(x)  # Output without activation function
        #x = F.relu(x)
        x = x.reshape(-1, 1)  # Reshape back to the original input shape using reshape
        return x

# class PositiveHuberLoss(nn.Module):
#     def __init__(self):
#         super(PositiveHuberLoss, self).__init__()
#         self.huber_loss = nn.HuberLoss()  # Definiere die MSE Loss Funktion

#     def forward(self, y_pred, y_true):
#         # Berechne den MSE
#         huber = self.huber_loss(y_pred, y_true)

#         # Bestrafe negative Vorhersagen, indem negative Werte mit ReLU bestraft werden
#         penalty = torch.mean(torch.relu(-y_pred))  # Penalty für negative Werte

#         # Kombiniere MSE und Penalty
#         total_loss = huber + penalty

#         return total_loss

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=8, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
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

# def save_models(epoch):
#     model_path = output_dir+r'Unetmodel'+str(id_lon)+'/'   
#     if not os.path.exists(model_path):
#         os.makedirs(model_path)
#     file = f"unet_epoch{epoch}.pth"
#     file_path = os.path.join(model_path, file)
#     torch.save(model.state_dict(),  file_path)
#     print("Checkpoint saved")

def train_model(id_point, epoch, train_loader, val_loader, device, model, criterion, optimizer):
    model = model.to(device)

    # initialize the early_stopping object
    model_path = Outfilepath+r'NNmodel_Point'+str(id_point)+'/'   
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    file = f"NN.pth"
    file_path = os.path.join(model_path, file)
    early_stopping = EarlyStopping(patience=8, verbose=False, path=file_path)

    # best_loss = 99999.

    for e in range(epoch):
        train_loss = 0.0
        val_loss = 0.0
        train_mae = 0.0
        val_mae = 0.0
        mae = nn.L1Loss() # mean absolute error
        train_loader = tqdm(train_loader)  #转换成tqdm类型 以方便增加日志的输出
        train_loader.set_description(f'[Train Epoch:{e+1:04d}/{epoch:04d} lr:{get_lr(optimizer):.6f}]')
        model.train() 
        # 训练集训练模型,对每个batch循环:
        for (i, (x,y)) in enumerate(train_loader,0):  # 0是下标起始位置默认为0
            inputs, labels = x.to(device), y.to(device)
            # 初始为0，清除上个batch的梯度信息
            optimizer.zero_grad()
            # 网络向前运行
            outputs = model(inputs)#.squeeze()
            # 计算loss值
            loss = criterion(outputs, labels)
            # 反向传播梯度 Compute gradients
            loss.backward()
            # 更新权重
            optimizer.step()
            train_loss += loss.item()
            train_mae += mae(outputs, labels).item()
            # 计算模型在一个epoch上的损失
            postfix = {
                'train_loss': f'{train_loss / (i + 1):.6f}',
                'train_mae': f'{train_mae / (i + 1):.6f}',
                }
            train_loader.set_postfix(log=postfix)
        
        # 验证集检验模型是否过拟合
        model.eval()  # 切换到评估模式
        with torch.no_grad():  # 禁用梯度计算
            val_loader = tqdm(val_loader) # 转换成tqdm类型 以方便增加日志的输出
            val_loader.set_description(f'[val Epoch:{e+1:04d}/{epoch:04d}]')
            for (i, (x,y)) in enumerate(val_loader, 0):
                inputs, labels = x.to(device), y.to(device)
                outputs = model(inputs)#.squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_mae += mae(outputs, labels).item()
                # 计算模型在一个epoch上的损失
                postfix = {
                    'val_loss': f'{val_loss / (i + 1):.6f}',
                    'val_mae': f'{val_mae / (i + 1):.6f}',
                    }
                val_loader.set_postfix(log=postfix)
            val_loss_epoch = val_loss / (i + 1)

        # Early stopping ; 保存模型
        # if test_loss_epoch < best_loss:
        #     save_models(e + 1)
        #     best_loss = test_loss_epoch
            
        early_stopping(val_loss_epoch, model)
        if early_stopping.early_stop:
            print("Early stopping!")
            break

    print('Finished Training!')

def evaluate_model(test_loader, device, model, scaler_y):
    model.eval()  # 设置模型为评估模式 sets the PyTorch model to evaluation mode, disabling operations like dropout, useful for inference and testing.
    predictions = []
    true_values = []

    with torch.no_grad():
        for (x, y) in tqdm(test_loader, desc="Testing"):
            inputs, labels = x.to(device), y.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            true_values.append(labels.cpu().numpy())

    # 将预测值和真实值转换为 NumPy 数组
    predictions1 = np.concatenate(predictions, axis=0)
    true_values1 = np.concatenate(true_values, axis=0)

    # 反标准化预测值和真实值
    Ozone_pred = scaler_y.inverse_transform(predictions1.reshape(-1, 1))
    Ozone_true = scaler_y.inverse_transform(true_values1.reshape(-1, 1)) # inverse_transform 输入为二维数组形状为 (n_samples, n_features)
    print(np.shape(Ozone_pred))
    print(np.shape(Ozone_true))

    # 计算 R² 分数
    r2 = R2_score(Ozone_pred, Ozone_true)
    print(f"R² Score: {r2}")
    return r2


# class LogCoshLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, y_t, y_prime_t):
#         ey_t = y_t - y_prime_t
#         return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))

# criterion = LogCoshLoss() # 'logcosh' works mostly like the mean squared error,
# # but will not be so strongly affected by the occasional wildly incorrect prediction.


#%% load data
cfile = pd.read_excel("./ChoosingPoints.xlsx",usecols="A:C")
lev_ls = np.array(cfile["Hybrid_ht"])
lat_ls = np.array(cfile["latitude"])
lon_ls = np.array(cfile["longitude"])
n_sample = len(lev_ls)
lon_list = [0.9,57.2,96.6,102.2,186.6,209.1,248.4,282.2,315.9,332.8]

### Ozone
filepath = "/hkfs/work/workspace/scratch/ou4895-ukesm2/data/"
ozone_file = xr.open_dataset(filepath+'data_ozone_50years_UKESM.nc',
                                  decode_times=False,engine="netcdf4").sel(longitude=lon_list, method="nearest") # 1960-1-1 to 1969-12-30
Ozone_train = ozone_file['field2101'][1:nt_train,:76,:,:]
Ozone_val = ozone_file['field2101'][nt_train+1:nt_train+nt_val,:76,:,:]
Ozone_test = ozone_file['field2101'][1-nt_test:,:76,:,:]
del ozone_file
print("end of loading ozone data")

oz_train_ls = []; oz_val_ls=[]; oz_test_ls = []
for i in range(n_sample):#range(n_sample):
    oz_train_ls.append(Ozone_train.sel(hybrid_ht=lev_ls[i],latitude=lat_ls[i],longitude=lon_ls[i], method="nearest"))
    oz_val_ls.append(Ozone_val.sel(hybrid_ht=lev_ls[i],latitude=lat_ls[i],longitude=lon_ls[i], method="nearest"))
    oz_test_ls.append(Ozone_test.sel(hybrid_ht=lev_ls[i],latitude=lat_ls[i],longitude=lon_ls[i], method="nearest"))


### Temp
temp_file = xr.open_dataset(filepath+'data_temp_50years_UKESM.nc',
                            decode_times=False).sel(longitude=lon_list, method="nearest") # 1960-1-1 to 2009-12-30
Temp_train=temp_file['temp'][:nt_train-1,:76,:,:]
Temp_val=temp_file['temp'][nt_train:nt_train+nt_val-1,:76,:,:]
Temp_test=temp_file['temp'][-nt_test:-1,:76,:,:]
del temp_file
print("end of loading temp data")

tp_train_ls = []; tp_val_ls = []; tp_test_ls = []
for i in range(n_sample):
    tp_train_ls.append(Temp_train.sel(latitude=lat_ls[i],longitude=lon_ls[i], method="nearest"))
    tp_val_ls.append(Temp_val.sel(latitude=lat_ls[i],longitude=lon_ls[i], method="nearest"))
    tp_test_ls.append(Temp_test.sel(latitude=lat_ls[i],longitude=lon_ls[i], method="nearest"))


#%% train and test
# 创建模型
criterion = nn.L1Loss()

r2_test = []
for id_point in range(n_sample):
    # 创建训练集、验证集和测试集
    X_train_data = xr.concat([tp_train_ls[id_point]], dim="variable").transpose("t", "variable", "hybrid_ht")
    Y_train_data = xr.concat([oz_train_ls[id_point]], dim="variable").transpose("t", "variable")
    X_validation_data = xr.concat([tp_val_ls[id_point]], dim="variable").transpose("t", "variable", "hybrid_ht")
    Y_validation_data = xr.concat([oz_val_ls[id_point]], dim="variable").transpose("t", "variable")
    X_test_data = xr.concat([tp_test_ls[id_point]], dim="variable").transpose("t", "variable", "hybrid_ht")
    Y_test_data = xr.concat([oz_test_ls[id_point]], dim="variable").transpose("t", "variable")

    X_train_std,X_validation_std,X_test_std,scaler_x = Standard(X_train_data,X_validation_data,X_test_data)
    Y_train_std,Y_validation_std,Y_test_std,scaler_y = Standard(Y_train_data,Y_validation_data,Y_test_data)

    X_train = torch.tensor(X_train_std, dtype=torch.float32).to(device)# (time, height)
    X_val = torch.tensor(X_validation_std, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test_std, dtype=torch.float32).to(device)
    Y_train = torch.tensor(Y_train_std, dtype=torch.float32).to(device)
    Y_val = torch.tensor(Y_validation_std, dtype=torch.float32).to(device)
    Y_test = torch.tensor(Y_test_std, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    test_dataset = TensorDataset(X_test, Y_test)

    # 查看创建训练&验证数据
    print(f'train num: {train_dataset.__len__()}, val num: {val_dataset.__len__()}, test num: {test_dataset.__len__()}')

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False,worker_init_fn=seed_worker(seed),generator=g)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False,worker_init_fn=seed_worker(seed),generator=g)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False,worker_init_fn=seed_worker(seed),generator=g)

    model = FeedForwardNN(input_shape=(1,76), num_hidden_layers=n_hidden_layers, hidden_layer_size=hidden_layer_size,dropout_prob=dropout_prob).to(device)
    # summary(model, input_size=(1, 76))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training
    train_model(id_point, 30, train_loader, val_loader, device, model, criterion, optimizer)
    # Testing
    r2_test.append(evaluate_model(test_loader, device, model, scaler_y))


# print out to excel file
X_point = np.arange(0, n_sample, 1)
df = pd.DataFrame(r2_test,index=X_point, columns=['R2_test_scores'])
df.index.name = "Points"
df.to_excel(Outfilepath+"R2_test_scores_NN_Offline_32yr_train_8yr_test.xlsx")


print("Finished!!")



# %%
