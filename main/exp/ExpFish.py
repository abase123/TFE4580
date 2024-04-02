
#torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
#Libaries for calculation and processing
import numpy as np
#libaries for data importng, formating and handling 
import pandas as pd
#For analysis and plotting
#others
import os
import time
import warnings
warnings.filterwarnings('ignore')

from .ExpBasic import ExpBasic
from utils.Metrics import metric



class Expfish(ExpBasic):
    def __init__(self, model, data_loader_train, data_loader_val, data_loader_test,device,criterion,optimizer,num_epochs):
        self.device = device
        self.model = model
        self.model = self.model.to(self.device)
        
        self.data_loader_train = data_loader_train
        self.data_loader_val = data_loader_val
        self.data_loader_test = data_loader_test
        
        
        self.criterion = criterion
        self.optim = optimizer
        self.num_epochs=num_epochs
        
        self.model_path = "crossModel"
        os.makedirs(self.model_path, exist_ok=True)
        
        
    def vali(self,vali_loader):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                pred, true = self._process_one_batch(batch_x, batch_y)
                loss = self.criterion(pred.detach().cpu(), true.detach().cpu())
                total_loss.append(loss.detach().item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self):
        train_loader = self.data_loader_train
        vali_loader = self.data_loader_val
        test_loader = self.data_loader_test

        train_steps = len(train_loader)
        model_optim = self.optim
        criterion = self.criterion

        for epoch in range(self.num_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            for block in self.model.encoder.encode_blocks:
                for layer in block.encode_layers:
                    layer.reset_attention_across_channels_details()
                    
                    
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                pred, true = self._process_one_batch(batch_x, batch_y)
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                loss.backward()
                model_optim.step()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            #vali_loss = self.vali(vali_loader)
            vali_loss = 0
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            
        torch.save(self.model.state_dict(), os.path.join(self.model_path, 'best_model.pth'))
        best_model_path = os.path.join(self.model_path, 'best_model.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, save_pred=False ):
        test_loader = self.data_loader_val
        self.model.eval()
        preds = []
        trues = []
        metrics_all = []
        instance_num = 0
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                pred, true = self._process_one_batch(batch_x, batch_y)
                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)
                if save_pred:
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())
                break
                
        metrics_all = np.stack(metrics_all, axis=0)
        metrics_mean = metrics_all.sum(axis=0) / instance_num
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        mae, mse, rmse, mape, mspe = metrics_mean
        print('mse:{}, mae:{}'.format(mse, mae))
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        if save_pred:
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)
        return

    def _process_one_batch(self, batch_x, batch_y):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        outputs = self.model(batch_x)
        return outputs, batch_y

    def eval(self, setting, save_pred=False):
        test_loader = self.data_loader_val
        self.model.eval()
        preds = []
        trues = []
        metrics_all = []
        instance_num = 0
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                pred, true = self._process_one_batch(batch_x, batch_y)
                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)
                
                if save_pred:
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())
                    
                break
            
        metrics_all = np.stack(metrics_all, axis=0)
        metrics_mean = metrics_all.sum(axis=0) / instance_num
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        mae, mse, rmse, mape, mspe = metrics_mean
        print('mse:{}, mae:{}'.format(mse, mae))
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        if save_pred:
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)