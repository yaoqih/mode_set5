import torch
from tqdm import tqdm
import os
import pandas as pd
from model import iTransformer
from data_provider.data_factory import data_provider
import numpy as np
import random
fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
class Config:
    pred_len = 2
    label_len = 0
    seq_len = 36
    root_path = './factor/'
    output_attention = False
    use_norm = True
    d_model = 512
    embed = 'timeF'
    dropout = 0.1
    d_ff = 1024
    e_layers=4
    factor = 1
    activation = 'gelu'
    n_heads = 8
    features = 'MS'
    target = 'open'
    inverse = False
    batch_size = 512
    freq = 'd'
    data = 'stock'
    num_workers = 0
    class_strategy='projection'
config=Config()
def predict_latest(load=True):
    checkpoint_name='checkpoint_new'
    model_path='idstock_96_2_modeliTransformer_datastock_sl36_pl2_dm512_nh8_el4_df1024_fc1_ebtimeF_desExp_csprojection_dp0.1_lr0.0001_lofinal_rate_mse_mixed_ftall_fl0.000796_fcTrue_fe4_fd0.017831'
    device =  torch.device('cuda:{}'.format(1))
    model= iTransformer.Model(config).float().to(device)
    if load:
        path = os.path.join('./checkpoints/', model_path)
        best_model_path = path + '/' + f'{checkpoint_name}.pth'
        if torch.load(best_model_path).get('net'):
            model.load_state_dict(torch.load(best_model_path)['net'])
        else:
            model.load_state_dict(torch.load(best_model_path))
    predict_path = './'
    model.eval()
    f=open(predict_path+f"fininal_result.csv", 'w')
    f.write('stock,date,score1,score1_true,score2,score2_ture\n')
    f.close()
    for day in range(2,200):
        pbar = tqdm(os.listdir(config.root_path))
        for file in pbar:
            pbar.set_description("Processing %s" % file) # 设置描述
            config.data_path = file
            pred_data, data_loader = data_provider(config, flag='final',last_n=day)
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
                    batch_x = batch_x.float().to(device)
                    batch_y = batch_y.float()
                    batch_x_mark = batch_x_mark.float().to(device)
                    batch_y_mark = batch_y_mark.float().to(device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -config.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :config.label_len, :], dec_inp], dim=1).float().to(device)
                    # encoder - decoder
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs = outputs.detach().cpu().numpy()
                    f_dim = -1 if config.features == 'MS' else 0
                    batch_y = batch_y.detach().cpu().numpy()
                    shape = outputs.shape
                    batch_y_shape = batch_y.shape
                    outputs = pred_data.inverse_transform(outputs.reshape((shape[0] * shape[1],) + shape[2:])).reshape(shape)
                    batch_y = pred_data.inverse_transform(batch_y.reshape((batch_y_shape[0] * batch_y_shape[1],) + batch_y_shape[2:])).reshape(batch_y_shape)
                    outputs = outputs[:, -config.pred_len:, f_dim:]
                    batch_y = batch_y[:, -config.pred_len:, f_dim:]
                    with open(predict_path+f"fininal_result.csv", 'a') as f:
                        for batch in range(len(outputs)):
                            f.write("'"+str(file.split('.')[0])[2:] + ',')
                            f.write(str(pred_data.latest_date))
                            for i in range(outputs.shape[1]):
                                f.write(','+str(outputs[batch][i][0])+','+str(batch_y[batch][i][0]))
                            f.write('\n')
    file_name=predict_path+f"fininal_result.csv"
    save_path=f"result2/"
    df=pd.read_csv(file_name)
    df['corr1']=np.abs((df['score1_true']-df['score1'])/df['score1'])
    df['corr2']=np.abs(df['score2_ture']-df['score2'])/df['score2']
    df['pred_rate']=(df['score2']-df['score1'])/df['score1']
    df['true_rate']=(df['score2_ture']-df['score1_true'])/df['score1_true']
    df['average_corr']=np.abs(df['pred_rate']+df['true_rate'])
    df.sort_values(by=['pred_rate'], inplace=True,ascending=False)
    # df['pred_rate'] = df['pred_rate'].round(3)
    date=df['date'].iloc[0]
    df.to_csv(save_path+f"{date}.csv",index=False)
    df=pd.read_csv(save_path+f"{date}.csv")
    code_list=pd.read_csv('./other_files/stock_list.csv')
    # df['date']="'"+df['date']
    df['name']=""
    for i in range(len(df)):
        df['name'].iloc[i]=code_list[code_list['code']==df['stock'].iloc[i]]['Name'].values[0].strip()
    df.to_csv(save_path+f"{date}.csv",index=False)

    return
predict_latest()