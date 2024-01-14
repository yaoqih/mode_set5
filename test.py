# factor=['SMA','EMA','MSTD','MVAR','RSV','RSI','KDJ','Bolling','MACD','CR','WR','CCI','TR','ATR','DMA','DMI','+DI','-DI','ADX','ADXR','TRIX','TEMA','VR','MFI','VWMA','CHOP','KER','KAMA','PPO','StochRSI','WT','Supertrend','Aroon','Z','AO','BOP','MAD','ROC','Coppock','Ichimoku','CTI','LRMA','ERI','FTR','RVGI','Inertia','KST','PGO','PSL','PVO','QQE']
# text=open('指标计算.py','r',encoding='utf-8').read()
# for f in factor:
#     if f not in text:
#         print(f)
import pandas as pd
df=pd.read_parquet("./other_files/SH600601.parquet").iloc[:,1:] 
df.drop(['AVGPRICE','STOCHslowd','ROCR','+DI','APO','WCLPRICE','MACDmacdsignal','CMO','SMA','ROC','MACDFIXmacd','ROCR100','MACDmacdhist','WR','-DI','Bolling','MEDPRICE','TSF'],inplace=True,axis=1)
res=[]
for method in ['pearson','spearman','kendall']:
    result=df.corr(method=method)
    temp=set()
    for col in range(result.shape[1]):
        for  row in range(result.shape[0]):
            if col!=row and result.iloc[col,row]>0.98 or result.iloc[col,row]<-0.98:
                temp.add(df.columns[col])
    res.append(temp)
print(res[0]&res[1]&res[2])
                
# print(df.corr(method='pearson'))
# print(df.corr(method='spearman'))
# print(df.corr(method='kendall'))

