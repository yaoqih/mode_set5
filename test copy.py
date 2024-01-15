import torch
import torch.nn as nn
import torch.nn.init as init
import torch
from tqdm import trange
from random import random,randint
import time
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv("predict_result/stock_96_96_iTransformer_stock_MS_ft36_sl0_ll2_pl512_dm8_nh4_el1_dl1024_df1_fctimeF_ebTrue_dtExp_projection_0/checkpoint.csv")
df['corr1_rate']=np.abs(df['pred1']-df['true1'])/df['true1']
df['corr2_rate']=np.abs(df['pred2']-df['true2'])/df['true2']
df['true_rate']=(df['true2']-df['true1'])/df['true1']
df['pred_rate']=(df['pred2']-df['pred1'])/df['pred1']
df.to_csv("predict_result/stock_96_96_iTransformer_stock_MS_ft36_sl0_ll2_pl512_dm8_nh4_el1_dl1024_df1_fctimeF_ebTrue_dtExp_projection_0/checkpoint_fixed.csv")
for rate in range(1,11):
    vaild_loss=df['valiloss'].quantile(rate/100) 
    print(vaild_loss,end='\t\t')
    select=df[df['valiloss']<vaild_loss]
    select.sort_values(by='pred_rate',ascending=True,inplace=True)
    for i in [10,20,30,40,50,60,70,80,90,100]:
        print(round(select['true_rate'].iloc[:i].mean(),3),end='\t')
    print()

# with trange(10) as t:
#   for i in t:
#     #设置进度条左边显示的信息
#     t.set_description("GEN %i"%i)
#     #设置进度条右边显示的信息
#     t.set_postfix(loss=random(),gen=randint(1,999),str="h",lst=[1,2])
#     time.sleep(0.1)
# pred=100
# n_d=1
# batch_size=1
# # 创建一个从1到3的向量
# column_vector = pred*10/torch.pow(torch.arange(pred,0,-1).view(-1, 1),2)

# # 重复该列向量在第二维度上扩展三次
# repeated_columns = column_vector.repeat(1, n_d)

# # 重复整个2D数组五次来创建3D数组
# repeated_3d_matrix = repeated_columns.unsqueeze(0).repeat(batch_size, 1, 1)

# print(repeated_3d_matrix)

# print(torch.linspace(1, 2, 96))
# class MyModel(nn.Module):
#     def __init__(self, input_features, output_features, feature_importances):
#         super(MyModel, self).__init__()
#         self.linear = nn.Linear(input_features, output_features)
        
#         # 初始化权重
#         self.init_weights(feature_importances)

#     def init_weights(self, feature_importances):
#         # 根据特征重要性计算初始权重
#         # 假设 feature_importances 是一个包含特征重要性评分的 PyTorch Tensor
#         weights = self.linear.weight.data

#         # 以某种方式将特征重要性映射到初始权重的范围
#         # 例如，你可以通过线性缩放的方式将重要性映射到一个合适的权重初始范围
#         # 以下为简单示例：将重要性乘以一个常数因子
#         weights = weights * feature_importances.unsqueeze(1)

#         # 设置权重（这里只设置了权重，没有设置偏置）
#         self.linear.weight = nn.Parameter(weights)

#     def forward(self, x):
#         return self.linear(x)

# # 假设的特征重要性评分，应与输入特征数量相同
# feature_importances = torch.tensor([0.6, 0.3, 0.1])

# # 创建模型实例
# model = MyModel(input_features=3, output_features=1, feature_importances=feature_importances)

# print(model.linear.weight)