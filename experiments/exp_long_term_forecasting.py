from data_provider.data_factory import data_provider,dataloader_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from torch.optim import lr_scheduler 
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
from transformers import AdamW, get_linear_schedule_with_warmup
import os
import time
import warnings
import numpy as np
import torch.nn.functional as F
from datetime import datetime
warnings.filterwarnings('ignore')
import random
from tqdm import tqdm
import wandb
from torch.utils.data import ConcatDataset
import pandas as pd
from safetensors.torch import load_model, save_model

class FinalRateMse(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return F.mse_loss((x[:,1,:]-x[:,0,:]),(y[:,1,:]-y[:,0,:]))
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return 0.5 * F.mse_loss(x, y) + 0.5 * F.l1_loss(x, y)
def print_architecture(model):
    name = type(model).__name__
    result = '-------------------%s---------------------\n' % name
    total_num_params = 0
    for i, (name, child) in enumerate(model.named_children()):
        num_params = sum([p.numel() for p in child.parameters()])
        total_num_params += num_params
        for i, (name, grandchild) in enumerate(child.named_children()):
            num_params = sum([p.numel() for p in grandchild.parameters()])
    result += '[Network %s] Total number of parameters : %.3f M\n' % (name, total_num_params / (1024*1024))
    result += '-----------------------------------------------\n'
    print(result)
class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        # model =torch.compile(model)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self,prarameter=False):
        if not prarameter:
            # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
            model_optim = AdamW(self.model.parameters(), lr=self.args.learning_rate)
        else:
            model_optim = AdamW(prarameter, lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self,asign=None):
        # criterion = CustomLoss()
        # criterion = nn.HuberLoss(delta=0.02)
        if not asign:
            if self.args.loss == 'mse':
                criterion = nn.MSELoss()
            elif self.args.loss == 'final_rate_mse':
                criterion = FinalRateMse()
            elif self.args.loss == 'mae':
                criterion = nn.L1Loss()
            elif self.args.loss == 'huber':
                # criterion = nn.SmoothL1Loss()
                criterion = nn.HuberLoss(delta=self.args.huber_delta)
            else:
                raise NotImplementedError
        else:
            if asign == 'mse':
                criterion = nn.MSELoss()
            elif asign == 'mae':
                criterion = nn.L1Loss()
            elif asign == 'huber':
                # criterion = nn.SmoothL1Loss()
                criterion = nn.HuberLoss(delta=self.args.huber_delta)
            elif asign == 'final_rate_mse':
                criterion = FinalRateMse()
            else:
                raise NotImplementedError
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting=None):
        # wandb config
        if self.args.use_wandb and setting:
            self.recoder=wandb.init(project="iTransformer_first_full",config=vars(self.args),resume=self.args.resume,save_code=True)
            # self.recoder=wandb.init(project="iTransformer_first_full",config=vars(args),resume='automatically',id='xemfgruk',save_code=True)
        if  not setting:
            self.recoder=wandb.init(project="iTransformer_first_full",resume=self.args.resume,save_code=True)
            self.args.d_model = self.recoder.config.d_model
            self.args.n_heads = self.recoder.config.n_heads
            self.args.e_layers = self.recoder.config.e_layers
            self.args.d_ff = self.recoder.config.d_ff
            self.args.dropout = self.recoder.config.dropout
            self.args.learning_rate = self.recoder.config.learning_rate
            self.model = self._build_model().to(self.device)
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                    self.args.model_id,
                    self.args.model,
                    self.args.data,
                    self.args.features,
                    self.args.seq_len,
                    self.args.label_len,
                    self.args.pred_len,
                    self.args.d_model,
                    self.args.n_heads,
                    self.args.e_layers,
                    self.args.d_layers,
                    self.args.d_ff,
                    self.args.factor,
                    self.args.embed,
                    self.args.distil,
                    self.args.des,
                    self.args.class_strategy, 0)
        
        # mkdir path
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
            
        # load data
        data_sets=[]
        print("loading data....")
        pbar = tqdm(os.listdir(self.args.root_path))
        for file in pbar:
            self.args.data_path = file
            train_data, train_loader = self._get_data(flag='train')
            data_sets.append(train_data)
        train_loader= dataloader_provider(self.args, flag='train',data_set=ConcatDataset(data_sets))
        print("data loaded....")
        
        # parameter initial
        rete=128/self.args.batch_size
        train_steps = len(os.listdir(self.args.root_path))
        warm_up_ratio = 0.1 # 定义要预热的step
        total_steps = int(train_loader.dataset.cumulative_sizes[-1] // self.args.batch_size+1) * self.args.train_epochs 
        train_loss = []
        process = tqdm(total=total_steps)
        min_loss=99999999999
        iter_count = 0
        
        # optim and schedule
        model_optim = self._select_optimizer()
        scheduler = get_linear_schedule_with_warmup(model_optim, num_warmup_steps = warm_up_ratio * total_steps, num_training_steps = total_steps)
        # scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
        #                                     steps_per_epoch = (train_steps*4500 // self.args.batch_size),
        #                                     pct_start = self.args.pct_start,
        #                                     epochs = self.args.train_epochs,
        #                                     max_lr = self.args.learning_rate)
        # early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        criterion = self._select_criterion()
        
        if self.args.resume:
            path_checkpoint = path + '/' +f'{self.args.checkpoint_name}.pth'  # 断点路径
            checkpoint = torch.load(path_checkpoint)  # 加载断点
            self.model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
            model_optim.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
            iter_count = checkpoint['iter_count']  # 设置开始的epoch
            scheduler.load_state_dict(checkpoint['lr_schedule'])#加载lr_scheduler

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        for epoch in range(self.args.train_epochs):
                # self.args.data_path = file
                # train_data, train_loader = self._get_data(flag='train')
                # vali_data, vali_loader = self._get_data(flag='val')
                # test_data, test_loader = self._get_data(flag='test')
            self.model.train()
            # epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                if self.args.resume:
                    if iter_count < checkpoint['iter_count']+1:
                        process.update(1)
                        continue
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                
                process.update(1)
                process.set_postfix(loss=loss.item())
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    scheduler.step()
                    # upload train_loss
                    if self.args.use_wandb :
                        if rete>=1 and iter_count%rete==0:
                            self.recoder.log({"train_loss":loss.item()},step=int(iter_count/rete))
                        else:
                            for i in range(int(1/rete)):
                                self.recoder.log({"train_loss":loss.item()},step=int(1/rete*iter_count+i))              
                if iter_count%int(500*rete)==0:
                    train_loss = np.average(train_loss)
                    test_sample_rate=0.1
                    test_data_sets=[]
                    val_data_sets=[]
                    print("testing....")
                    for file in tqdm(range(int(len(os.listdir(self.args.root_path))*test_sample_rate))):
                        self.args.data_path = random.choice(os.listdir(self.args.root_path))
                        test_data, test_loader = self._get_data(flag='test')
                        val_data, val_loader = self._get_data(flag='val')
                        test_data_sets.append(test_data)
                        val_data_sets.append(val_data)
                    criterion_test = self._select_criterion(asign='mse')
                    test_loader= dataloader_provider(self.args, flag='train_test',data_set=ConcatDataset(test_data_sets))
                    test_loss = self.vali(test_data_sets, test_loader, criterion_test)
                    val_loader= dataloader_provider(self.args, flag='train_test',data_set=ConcatDataset(val_data_sets))
                    vali_loss = self.vali(val_data_sets, val_loader, criterion_test)
                    checkpoint = {
                        "net": self.model.state_dict(),
                        'optimizer': model_optim.state_dict(),
                        "iter_count": iter_count,
                        'lr_schedule': scheduler.state_dict()
                    }
                    
                    # save module
                    if min_loss>vali_loss.item():
                        min_loss=vali_loss.item()
                        torch.save(checkpoint, path + '/' + f'{self.args.checkpoint_name}.pth')
                    torch.save(checkpoint, path + '/' + f'{self.args.checkpoint_name}_new.pth')
                    
                    if self.args.use_wandb:
                        self.recoder.log({"vali_loss":vali_loss,"test_loss":test_loss,"lr":model_optim.param_groups[0]["lr"]},step=int(iter_count/rete)+1)
                    print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                        epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                    train_loss = []
                
            # early_stopping(vali_loss, self.model, path)
            # if early_stopping.early_stop:
                # print("Early stopping")
                # break

            # adjust_learning_rate(model_optim, epoch + 1, self.args)
            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + f'{self.args.checkpoint_name}.pth'
        self.model.load_state_dict(torch.load(best_model_path)['net'])
        torch.save(self.model.state_dict(), path + '/' + f'{self.args.checkpoint_name}_new.pth')
        self.model.load_state_dict(torch.load(best_model_path)['net'])
        torch.save(self.model.state_dict(), path + '/' + f'{self.args.checkpoint_name}.pth')

        return self.model
    def pre_train_test(self,setting):
        
                # wandb config
        if self.args.use_wandb and setting:
            self.recoder=wandb.init(project="iTransformer_first_full",config=vars(self.args),resume=self.args.resume,save_code=True)
            # self.recoder=wandb.init(project="iTransformer_first_full",config=vars(args),resume='automatically',id='xemfgruk',save_code=True)
        if  not setting:
            self.recoder=wandb.init(project="iTransformer_first_full",resume=self.args.resume,save_code=True)
            self.args.d_model = self.recoder.config.d_model
            self.args.n_heads = self.recoder.config.n_heads
            self.args.e_layers = self.recoder.config.e_layers
            self.args.d_ff = self.recoder.config.d_ff
            self.args.dropout = self.recoder.config.dropout
            self.args.learning_rate = self.recoder.config.learning_rate
            self.model = self._build_model().to(self.device)
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                    self.args.model_id,
                    self.args.model,
                    self.args.data,
                    self.args.features,
                    self.args.seq_len,
                    self.args.label_len,
                    self.args.pred_len,
                    self.args.d_model,
                    self.args.n_heads,
                    self.args.e_layers,
                    self.args.d_layers,
                    self.args.d_ff,
                    self.args.factor,
                    self.args.embed,
                    self.args.distil,
                    self.args.des,
                    self.args.class_strategy, 0)
        
        # mkdir path
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
            
        # # load data
        # data_sets=[]
        # print("loading data....")
        # pbar = tqdm(os.listdir(self.args.root_path))
        # for file in pbar:
        #     self.args.data_path = file
        #     train_data, train_loader = self._get_data(flag='train')
        #     data_sets.append(train_data)
        # train_loader= dataloader_provider(self.args, flag='train',data_set=ConcatDataset(data_sets))
        # print("data loaded....")
        
        # parameter initial
        rete=128/self.args.batch_size
        train_steps = len(os.listdir(self.args.root_path))
        warm_up_ratio = 0.1 # 定义要预热的step
        # total_steps = int(train_loader.dataset.cumulative_sizes[-1] // self.args.batch_size+1) * self.args.train_epochs 
        train_loss = []
        min_loss=99999999999
        iter_count = 0
        
        if self.args.resume:
            path_checkpoint = path + '/' + f'{self.args.checkpoint_name}.pth'  # 断点路径
            checkpoint = torch.load(path_checkpoint)  # 加载断点
            self.model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
            model_optim.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
            iter_count = checkpoint['iter_count']  # 设置开始的epoch
            scheduler.load_state_dict(checkpoint['lr_schedule'])#加载lr_scheduler
            
        # optim and schedule
        # scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
        #                                     steps_per_epoch = (train_steps*4500 // self.args.batch_size),
        #                                     pct_start = self.args.pct_start,
        #                                     epochs = self.args.train_epochs,
        #                                     max_lr = self.args.learning_rate)
        # early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        pbar = tqdm(os.listdir(self.args.root_path))

        predict_path = './predict_result/' + setting + '/'
        if not os.path.exists(predict_path):
            os.makedirs(predict_path)
        f=open(predict_path+f"{self.args.checkpoint_name}.csv", 'w')
        f.write("stock,valiloss,true1,pred1,true2,pred2\n")
        f.close()
        f=open(predict_path+f"{self.args.checkpoint_name}_before.csv", 'w')
        f.write("stock,valiloss,true1,pred1,true2,pred2\n")
        f.close()
        for file in pbar:
            self.args.data_path = file
            train_data, train_loader = self._get_data(flag='train')
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')

            path_checkpoint = path + '/' + f'{self.args.checkpoint_name}.pth'  # 断点路径
            checkpoint = torch.load(path_checkpoint)  # 加载断点
            if checkpoint.get('net'):
                self.model.load_state_dict(checkpoint['net'])
            else:
                self.model.load_state_dict(checkpoint)

            criterion_test = self._select_criterion(asign='mse')
            test_loss_before = self.vali(test_data, test_loader, criterion_test)
            vali_loss_before = self.vali(vali_data, vali_loader, criterion_test)
            model_optim = self._select_optimizer()
            # model_optim = self._select_optimizer(self.model.projector.parameters())
            self.predict_single(setting,file_end='_before',vali_loss=vali_loss_before)
            total_steps=len(train_data)//self.args.batch_size
            # scheduler = get_linear_schedule_with_warmup(model_optim, num_warmup_steps = warm_up_ratio * total_steps, num_training_steps = total_steps)
            self.model.train()
            # epoch_time = time.time()
            for epoch in range(self.args.train_epochs+1):
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    iter_count += 1
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                        batch_x_mark = None
                        batch_y_mark = None
                    else:
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            loss = criterion(outputs, batch_y)
                            train_loss.append(loss.item())
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()
                        # scheduler.step()
                        # upload train_loss
                        if self.args.use_wandb :
                            if rete>=1 and iter_count%rete==0:
                                self.recoder.log({"train_loss":loss.item()},step=int(iter_count/rete))
                            else:
                                for i in range(int(1/rete)):
                                    self.recoder.log({"train_loss":loss.item()},step=int(1/rete*iter_count+i))              
                
                train_loss = np.average(train_loss)

                criterion_test = self._select_criterion(asign='mse')
                test_loss = self.vali(test_data, test_loader, criterion_test)
                vali_loss = self.vali(vali_data, vali_loader, criterion_test)
                # vali_loss_before=vali_loss
                # test_loss_before=test_loss
                if self.args.use_wandb:
                    self.recoder.log({"vali_loss":vali_loss,"test_loss":test_loss,"test_loss_before":vali_loss_before,"vali_loss_before":test_loss_before},step=int(iter_count/rete)+1)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                if epoch==self.args.train_epochs-1:
                    self.predict_single(setting,vali_loss=vali_loss)
                train_loss = []
                
            # early_stopping(vali_loss, self.model, path)
            # if early_stopping.early_stop:
                # print("Early stopping")
                # break

            # adjust_learning_rate(model_optim, epoch + 1, self.args)
            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)
        checkpoint_name=self.args.checkpoint_name
        df=pd.read_csv(predict_path+f"{checkpoint_name}_before.csv")
        df['corr1_rate']=np.abs(df['pred1']-df['true1'])/df['true1']
        df['corr2_rate']=np.abs(df['pred2']-df['true2'])/df['true2']
        df['true_rate']=(df['true2']-df['true1'])/df['true1']
        df['pred_rate']=(df['pred2']-df['pred1'])/df['pred1']
        df.to_csv(predict_path+f"{checkpoint_name}_before.csv")
        df=pd.read_csv(predict_path+f"{checkpoint_name}.csv")
        df['corr1_rate']=np.abs(df['pred1']-df['true1'])/df['true1']
        df['corr2_rate']=np.abs(df['pred2']-df['true2'])/df['true2']
        df['true_rate']=(df['true2']-df['true1'])/df['true1']
        df['pred_rate']=(df['pred2']-df['pred1'])/df['pred1']
        df.to_csv(predict_path+f"{checkpoint_name}.csv")
        return self.model
    def pre_train_predict(self,setting,save=False,load=False):
        
                # wandb config
        # if self.args.use_wandb and setting:
        #     self.recoder=wandb.init(project="iTransformer_first_full",config=vars(self.args),resume=self.args.resume,save_code=True)
            # self.recoder=wandb.init(project="iTransformer_first_full",config=vars(args),resume='automatically',id='xemfgruk',save_code=True)
        if  not setting:
            self.recoder=wandb.init(project="iTransformer_first_full",resume=self.args.resume,save_code=True)
            self.args.d_model = self.recoder.config.d_model
            self.args.n_heads = self.recoder.config.n_heads
            self.args.e_layers = self.recoder.config.e_layers
            self.args.d_ff = self.recoder.config.d_ff
            self.args.dropout = self.recoder.config.dropout
            self.args.learning_rate = self.recoder.config.learning_rate
            self.model = self._build_model().to(self.device)
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                    self.args.model_id,
                    self.args.model,
                    self.args.data,
                    self.args.features,
                    self.args.seq_len,
                    self.args.label_len,
                    self.args.pred_len,
                    self.args.d_model,
                    self.args.n_heads,
                    self.args.e_layers,
                    self.args.d_layers,
                    self.args.d_ff,
                    self.args.factor,
                    self.args.embed,
                    self.args.distil,
                    self.args.des,
                    self.args.class_strategy, 0)
        
        # mkdir path
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
            
        # # load data
        # data_sets=[]
        # print("loading data....")
        # pbar = tqdm(os.listdir(self.args.root_path))
        # for file in pbar:
        #     self.args.data_path = file
        #     train_data, train_loader = self._get_data(flag='train')
        #     data_sets.append(train_data)
        # train_loader= dataloader_provider(self.args, flag='train',data_set=ConcatDataset(data_sets))
        # print("data loaded....")
        
        # parameter initial
        rete=128/self.args.batch_size
        train_steps = len(os.listdir(self.args.root_path))
        warm_up_ratio = 0.1 # 定义要预热的step
        # total_steps = int(train_loader.dataset.cumulative_sizes[-1] // self.args.batch_size+1) * self.args.train_epochs 
        train_loss = []
        min_loss=99999999999
        iter_count = 0
        
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        pbar = tqdm(os.listdir(self.args.root_path))

        predict_path = './'
        if not os.path.exists(predict_path):
            os.makedirs(predict_path)
        f=open(predict_path+f"fininal_result.csv", 'w')
        f.write('stock,date,score1,score2\n')
        f.close()
        for file in pbar:
            self.args.data_path = file
            train_data, train_loader = self._get_data(flag='train')

            if load:
                load_model(self.model,f"{file.split('.')[0]}.safetensors")
            else:
                path_checkpoint = path + '/' + f'{self.args.checkpoint_name}_new.pth'  # 断点路径
                checkpoint = torch.load(path_checkpoint)  # 加载断点
                if checkpoint.get('net'):
                    self.model.load_state_dict(checkpoint['net'])
                else:
                    self.model.load_state_dict(checkpoint)
                model_optim = self._select_optimizer()
                self.model.train()
                # epoch_time = time.time()
                for epoch in range(self.args.train_epochs+1):
                    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                        iter_count += 1
                        model_optim.zero_grad()
                        batch_x = batch_x.float().to(self.device)
                        batch_y = batch_y.float().to(self.device)
                        if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                            batch_x_mark = None
                            batch_y_mark = None
                        else:
                            batch_x_mark = batch_x_mark.float().to(self.device)
                            batch_y_mark = batch_y_mark.float().to(self.device)

                        # decoder input
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                        # encoder - decoder
                        if self.args.use_amp:
                            with torch.cuda.amp.autocast():
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                                f_dim = -1 if self.args.features == 'MS' else 0
                                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                                loss = criterion(outputs, batch_y)
                                train_loss.append(loss.item())
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            loss = criterion(outputs, batch_y)
                            train_loss.append(loss.item())

                        if self.args.use_amp:
                            scaler.scale(loss).backward()
                            scaler.step(model_optim)
                            scaler.update()
                        else:
                            loss.backward()
                            model_optim.step()
                            # scheduler.step()
                            # upload train_loss
                            # if self.args.use_wandb :
                            #     if rete>=1 and iter_count%rete==0:
                            #         self.recoder.log({"train_loss":loss.item()},step=int(iter_count/rete))
                            #     else:
                            #         for i in range(int(1/rete)):
                            #             self.recoder.log({"train_loss":loss.item()},step=int(1/rete*iter_count+i))      
            if save:       
                save_model(self.model, f"{file.split('.')[0]}.safetensors")
            train_loss = np.average(train_loss)
            self.predict_single_least(setting)              
            train_loss = []
        self.final_result_deal(predict_path+f"fininal_result.csv",'./result2/')
       
        return self.model
    
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = path + '/' + f'{self.args.checkpoint_name}.pth'
        if test:
            print('loading model')
            if torch.load(best_model_path).get('net'):
                self.model.load_state_dict(torch.load(best_model_path)['net'])
            else:
                self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                # if i % 20 == 0:
                    # input = batch_x.detach().cpu().numpy()
                    # if test_data.scale and self.args.inverse:
                    #     shape = input.shape
                    #     input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    # gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    # pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        visual(trues.reshape(-1), preds.reshape(-1), os.path.join(folder_path, str(i) + '.pdf'))

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        with open('test_log.txt','a') as f:
            f.write(datetime.strftime(datetime.today(),'%Y-%m-%d %H:%M:%S')+'\n>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n'.format(setting))
            f.write('mse:{}, mae:{}\n'.format(mse, mae))
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return


    def predict(self, setting, load=False):
        checkpoint_name=self.args.checkpoint_name
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + f'{checkpoint_name}.pth'
            if torch.load(best_model_path).get('net'):
                self.model.load_state_dict(torch.load(best_model_path)['net'])
            else:
                self.model.load_state_dict(torch.load(best_model_path))
        preds = []
        trues = []
        predict_path = './predict_result/' + setting + '/'
        open(predict_path+f"{checkpoint_name}.csv", 'w').close()
        if not os.path.exists(predict_path):
            os.makedirs(predict_path)
        self.model.eval()
        pbar = tqdm(os.listdir(self.args.root_path))
        for file in pbar:
            pbar.set_description("Processing %s" % file) # 设置描述
            self.args.data_path = file
            pred_data, pred_loader = self._get_data(flag='pred')
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float()
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs = outputs.detach().cpu().numpy()
                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y.detach().cpu().numpy()
                    if pred_data.scale and self.args.inverse:
                        shape = outputs.shape
                        outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                        batch_y = pred_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.reshape((shape[0] * shape[1],) + shape[2:])).reshape(shape)
                    batch_y = pred_data.inverse_transform(batch_y.reshape((shape[0] * shape[1],) + shape[2:])).reshape(shape)
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    with open(predict_path+f"{checkpoint_name}.csv", 'a') as f:
                        for batch in range(len(batch_y)):
                            f.write(str(file) + ',')
                            for i in range(batch_y.shape[1]):
                                f.write(','+str(batch_y[batch][i][0]) + ','+str(outputs[batch][i][0]))
                            f.write('\n')
                    preds.append(outputs)
                    trues.append(batch_y)
        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        df=pd.read_csv(predict_path+f"{checkpoint_name}.csv")
        df['corr1_rate']=np.abs(df['pred1']-df['true1'])/df['true1']
        df['corr2_rate']=np.abs(df['pred2']-df['true2'])/df['true2']
        df['true_rate']=(df['true2']-df['true1'])/df['true1']
        df['pred_rate']=(df['pred2']-df['pred1'])/df['pred1']
        df.to_csv(predict_path+f"{checkpoint_name}.csv")

        # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        # np.save(folder_path + 'real_prediction.npy', preds)

        return
    def name_add(self,filename):
        df=pd.read_csv(filename)
        code_list=pd.read_csv('./other_files/stock_list.csv')
        # df['date']="'"+df['date']
        df['name']=""
        for i in range(len(df)):
            df['name'].iloc[i]=code_list[code_list['code']==df['stock'].iloc[i]]['Name'].values[0].strip()
        df.to_csv(filename,index=False)
    def predict_latest(self, setting, load=False):
        checkpoint_name=self.args.checkpoint_name
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + f'{checkpoint_name}.pth'
            if torch.load(best_model_path).get('net'):
                self.model.load_state_dict(torch.load(best_model_path)['net'])
            else:
                self.model.load_state_dict(torch.load(best_model_path))
        preds = []
        trues = []
        predict_path = './'
        f=open(predict_path+f"fininal_result.csv", 'w')
        f.write('stock,date,score1,score2\n')
        f.close()
        self.model.eval()
        pbar = tqdm(os.listdir(self.args.root_path))
        for file in pbar:
            pbar.set_description("Processing %s" % file) # 设置描述
            self.args.data_path = file
            pred_data, pred_loader = self._get_data(flag='final')
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float()
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs = outputs.detach().cpu().numpy()
                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y.detach().cpu().numpy()
                    if pred_data.scale and self.args.inverse:
                        shape = outputs.shape
                        outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                        batch_y = pred_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.reshape((shape[0] * shape[1],) + shape[2:])).reshape(shape)

                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    with open(predict_path+f"fininal_result.csv", 'a') as f:
                        for batch in range(len(outputs)):
                            f.write("'"+str(file.split('.')[0])[2:] + ',')
                            f.write(str(pred_data.latest_date))
                            for i in range(outputs.shape[1]):
                                f.write(','+str(outputs[batch][i][0]))
                            f.write('\n')

        self.final_result_deal(predict_path+f"fininal_result.csv",'./result/')
        return
    def final_result_deal(self,file_name,save_path):
        df=pd.read_csv(file_name)
        df['pred_rate']=(df['score2']-df['score1'])/df['score1']
        df.sort_values(by=['pred_rate'], inplace=True,ascending=False)
        df['pred_rate'] = df['pred_rate'].round(3)
        date=df['date'].iloc[0]
        df.to_csv(save_path+f"{date}.csv",index=False)
        self.name_add(save_path+f"{date}.csv")

    def predict_single(self, setting, load=False,file_end='',vali_loss=0):
        checkpoint_name=self.args.checkpoint_name
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + f'{checkpoint_name}.pth'
            if torch.load(best_model_path).get('net'):
                self.model.load_state_dict(torch.load(best_model_path)['net'])
            else:
                self.model.load_state_dict(torch.load(best_model_path))
        preds = []
        trues = []
        predict_path = './predict_result/' + setting + '/'
        self.model.eval()
        # pbar = tqdm(os.listdir(self.args.root_path))
        # for file in pbar:
            # pbar.set_description("Processing %s" % file) # 设置描述
            # self.args.data_path = file
        pred_data, pred_loader = self._get_data(flag='pred')
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = pred_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
                shape = outputs.shape
                outputs = pred_data.inverse_transform(outputs.reshape((shape[0] * shape[1],) + shape[2:])).reshape(shape)
                batch_y = pred_data.inverse_transform(batch_y.reshape((shape[0] * shape[1],) + shape[2:])).reshape(shape)
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                with open(predict_path+f"{checkpoint_name}{file_end}.csv", 'a') as f:
                    text=''
                    for batch in range(len(batch_y)):
                        text+=str(self.args.data_path)+','
                        text+=str(vali_loss)
                        for i in range(batch_y.shape[1]):
                            text+=','+str(batch_y[batch][i][0]) + ','+str(outputs[batch][i][0])
                        text+='\n'
                    f.write(text)
                preds.append(outputs)
                trues.append(batch_y)
  
        # preds = np.array(preds)
        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        # np.save(folder_path + 'real_prediction.npy', preds)

        return
    def predict_single_least(self, setting, load=False,file_end='',vali_loss=0):
        checkpoint_name=self.args.checkpoint_name
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + f'{checkpoint_name}.pth'
            if torch.load(best_model_path).get('net'):
                self.model.load_state_dict(torch.load(best_model_path)['net'])
            else:
                self.model.load_state_dict(torch.load(best_model_path))
        preds = []
        trues = []
        self.model.eval()
        # pbar = tqdm(os.listdir(self.args.root_path))
        # for file in pbar:
            # pbar.set_description("Processing %s" % file) # 设置描述
            # self.args.data_path = file
        pred_data, pred_loader = self._get_data(flag='final')
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = pred_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
                shape = outputs.shape
                outputs = pred_data.inverse_transform(outputs.reshape((shape[0] * shape[1],) + shape[2:])).reshape(shape)
                # batch_y = pred_data.inverse_transform(batch_y.reshape((shape[0] * shape[1],) + shape[2:])).reshape(shape)
                # batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                with open(f"fininal_result.csv", 'a') as f:
                        for batch in range(len(outputs)):
                            f.write(str(self.args.data_path.split('.')[0])[2:] + ',')
                            f.write(str(pred_data.latest_date))
                            for i in range(outputs.shape[1]):
                                f.write(','+str(outputs[batch][i][0]))
                            f.write('\n')
  
        # preds = np.array(preds)
        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        # np.save(folder_path + 'real_prediction.npy', preds)

        return