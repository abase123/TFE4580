
#torch
import torch.nn.functional as F
from torch.utils.data import Dataset
#Libaries for calculation and processing
from einops import rearrange, repeat
from sklearn.preprocessing import StandardScaler
#libaries for data importng, formating and handling 
import pandas as pd
#For analysis and plotting
import matplotlib.pyplot as plt
#others
import os
import warnings
warnings.filterwarnings('ignore')


class Dataset_MTS(Dataset):
    def __init__(self, root_path, data_path, flag, size, 
                data_split,scale, scale_statistic,stride):
        # size [seq_len, label_len, pred_len]
        # info
        self.in_len = size[0]
        self.out_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.scale = scale
        self.scaler = StandardScaler()
        #self.inverse = inverse
        self.df_raw = 0
        self.root_path = root_path
        self.data_path = data_path
        self.data_split = data_split
        self.scale_statistic = scale_statistic
        self.stride = stride
        
        self.total_windows = 0
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        self.df_raw = df_raw
    
        train_num = int(len(df_raw)*self.data_split[0]); 
        test_num = int(len(df_raw)*self.data_split[2])
        val_num = len(df_raw) - train_num - test_num;
         
        border1s = [0, train_num - self.in_len, train_num + val_num - self.in_len]
        border2s = [train_num, train_num+val_num, train_num + val_num + test_num]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        cols_data = df_raw.columns[6:]
        df_data = df_raw[cols_data]

        if self.scale:
            data = self.scaler.fit_transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
    
        self.total_windows = (len(self.data_x)- self.in_len - self.out_len) // self.stride + 1
    
    def __getitem__(self, index):
        
        s_begin = index * self.stride
        s_end = s_begin + self.in_len
        r_begin = s_end
        r_end = r_begin + self.out_len
        
        if r_end > len(self.data_x):
            s_begin = len(self.data_x) - self.in_len - self.out_len
            s_end = s_begin + self.in_len
            r_begin = s_end
            r_end = r_begin + self.out_len
        

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

       
        return seq_x, seq_y
    
    def __len__(self):
        return len(self.data_x) - self.in_len- self.out_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def get_raw_data_before_split(self):
        
        return self.df_raw

    def plot_cluster_split(self,cluster_index):
            # Number of features

        # Calculate split indices
        train_num = int(len(self.df_raw)*self.data_split[0])
        test_num = int(len(self.df_raw)*self.data_split[2])
        val_num = len(self.df_raw) - train_num - test_num
        train_end = train_num
        val_end = train_num + val_num

        plt.figure(figsize=(14, 6))
        
        # Plotting training data
        plt.plot(self.df_raw.iloc[:train_end, cluster_index], label='Training Data', color='blue')
        
        # Plotting validation data
        plt.plot(range(train_end, val_end), self.df_raw.iloc[train_end:val_end, cluster_index], label='Validation Data', color='red')
        
        # Plotting test data
        plt.plot(range(val_end, len(self.df_raw)), self.df_raw.iloc[val_end:, cluster_index], label='Test Data', color='green')
        
        plt.title(f'Feature {self.df_raw.columns[cluster_index]} Across Splits')
        plt.xlabel('Index')
        plt.ylabel('Feature Value')
        plt.legend()
        plt.show()

        
        
       
class Dataset_MTS_simplified(Dataset):
    def __init__(self,df_data,size,stride,cols_target):
        
        # size [seq_len, label_len, pred_len]
        # info
        self.in_len = size[0]
        self.out_len = size[1]
        self.cols_target = cols_target
        # init
        self.df_raw = df_data
        self.stride = stride
        self.total_windows = 0
        self.__read_data__()

    def __read_data__(self):
        
        data_len = len(self.df_raw)
        
        cols_data = self.df_raw.columns[0:]
        df_data = self.df_raw[cols_data]

        data = df_data.values

        self.data_x = data[0:data_len]
        self.data_y = data[0:data_len]
    
        self.total_windows = self.total_windows = ((data_len - self.in_len - self.out_len) // self.stride) + 1
    
    def __getitem__(self, index):
         
        s_begin = index * self.stride
        s_end = s_begin + self.in_len
        r_begin = s_end
        r_end = r_begin + self.out_len
        #r_begin = s_begin
        #r_end = r_begin + self.out_len
        
        if s_begin > len(self.data_x):
            s_begin = len(self.data_x) - self.in_len - self.out_len
            s_end = s_begin + self.in_len
            r_begin = s_end
            r_end = r_begin + self.out_len
        

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        print(f"{s_begin},{s_end}")
       
        return seq_x, seq_y
    
    def __len__(self):
        return self.total_windows
  
    def get_raw_data_before_split(self):
        return self.df_raw

 

        
        
       
    
    
    