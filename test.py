# factor=['SMA','EMA','MSTD','MVAR','RSV','RSI','KDJ','Bolling','MACD','CR','WR','CCI','TR','ATR','DMA','DMI','+DI','-DI','ADX','ADXR','TRIX','TEMA','VR','MFI','VWMA','CHOP','KER','KAMA','PPO','StochRSI','WT','Supertrend','Aroon','Z','AO','BOP','MAD','ROC','Coppock','Ichimoku','CTI','LRMA','ERI','FTR','RVGI','Inertia','KST','PGO','PSL','PVO','QQE']
# text=open('指标计算.py','r',encoding='utf-8').read()
# for f in factor:
#     if f not in text:
#         print(f)
import pandas as pd
df=pd.read_parquet("./factor/SH600601.parquet")
df.drop('date',inplace=True,axis=1)
print(df.corr(method='pearson'))
print(df.corr(method='spearman'))
print(df.corr(method='kendall'))