from stock_data_tools import last_train_data,deleteNullFile,delet,init,compare_date
import json
from urllib.request import urlopen
import requests
import os
import pandas as pd
import time
from multiprocessing import Pool,Lock
from config import basic_data_save_path,other_file_path
s=requests.session()
s.trust_env=False
def download_basic_data(inputfile):
    # try:
    trandtime=last_train_data()
    basic_data_save_path=inputfile.split(',')[1]
    inputfile=inputfile.split(',')[0]
    inputfile=str(inputfile)
    if len(inputfile)<6:
        inputfile='0'*(6-len(inputfile))+inputfile
    if inputfile[0]=='6':
        down_num='1.'+inputfile
        inputfile='SH'+inputfile+'.parquet'
    else:
        down_num='0.'+inputfile
        inputfile='SZ'+inputfile+'.parquet'
    if inputfile not in os.listdir(basic_data_save_path):
        # url='http://80.push2his.eastmoney.com/api/qt/stock/kline/get?&secid='+down_num+'&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&klt=101&fqt=1&beg=0&end=20500101'
        # url='http://68.push2his.eastmoney.com/api/qt/stock/kline/get?&secid='+down_num+'&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&klt=101&fqt=0&end=20500101&lmt=1000000'
        url ='http://push2his.eastmoney.com/api/qt/stock/kline/get?&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&klt=101&fqt=2&secid='+down_num+'&beg=0&end=20500000'
        # url ='http://push2his.eastmoney.com/api/qt/stock/kline/get?&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&klt=101&fqt=2&secid=1.600000&beg=0&end=20500000&_=1606616431926'
        # content=urlopen(url).read()
        content=s.get(url).content
        if len(content)==0:
           content=s.get(url).content
        if len(content)==0:
            content=s.get(url).content
        if len(content)==0:
            print('HTTP,Error '+inputfile)
            return -1
        content=content.decode('utf-8','ignore').replace('\n','')
        content=json.loads(content)
        # lock.acquire()
        f = open(basic_data_save_path+inputfile,'a',encoding='utf-8')
        f.write('date,open,close,high,low,volume,amount,amplitude,pct_chg,change,turnover_rate\n')
        f.write('\n'.join(content['data']['klines']))
        f.close()
        df=pd.read_csv(basic_data_save_path+inputfile,encoding='utf-8')
        # # lock.release()
        # df=df[~(df['pct_chg'].isin([None]))]
        # df=df[~(df['turnover_rate'].isin([None]))]
        # df=df[~(df['turnover_rate'].isin(['None']))]
        # df=df[~(df['open'].isin([0.0]))]
        # df=df[~(df['open'].isin([0]))]
        # df=df[~(df['名称'].isin(['']))]
        # df['turnover_rate']=pd.DataFrame(df['turnover_rate'],dtype=np.float)
        # df.rename(columns={'日期':'date','前收盘':'前close'},inplace=True)
        # df.sort_values(by=['date'], inplace=True)
        # df.drop(['change'],axis=1,inplace=True)
        # df.drop(['amount'],axis=1,inplace=True)
        # df.drop(['pct_chg'],axis=1,inplace=True)
        df.to_parquet(basic_data_save_path+inputfile)
        if df.shape[0]>0 and len(df)>50 and compare_date(df['date'].iloc[-1],trandtime):
            time.sleep(0.01)
        # else:
        #     if inputfile in os.listdir(basic_data_save_path):
        #         os.remove(basic_data_save_path+inputfile)
    # except:
    #     if inputfile in os.listdir(basic_data_save_path):
    #         os.remove(basic_data_save_path+inputfile)
    #     print(inputfile)
    #     print('error')
def callback(result):
    # 在回调函数中关闭和加入进程池
    global lock
    with lock:
        print("Process finished with result:", result)
def download_basic_data_all(data_path,deubg=False):
    '''
    下载所有原始数据
    :data_path:存放地址
    '''
    if 'SZ000001.parquet' not in os.listdir(data_path)  or pd.read_parquet(data_path+'SZ000001.parquet')['date'].iat[-1]!=last_train_data():
        delet(data_path)
    df = pd.read_csv(other_file_path+'stock_list.csv',encoding='utf-8')['公司代码 ']
    codelist = list(df)
    download_basic_data('1'+','+data_path)
    datetime_now=pd.read_parquet(data_path + 'SZ000001.parquet')['date'].iloc[-1]
    for i in range(1,len(codelist)):
        codelist[i]=str(codelist[i])+','+data_path
    # for i in codelist:
    #     download_basic_data(i)
    if deubg:
        for code in codelist[1:]:
            download_basic_data(code)
    else:
        lock = Lock()
        with Pool(16, initializer=init, initargs=(lock,)) as pool:
            # 使用 map_async，并传递回调函数
            result = pool.map_async(download_basic_data, codelist[1:], callback=callback)
            
            # 等待所有子进程完成
            result.wait()

    print("All processes are finished.")
    deleteNullFile(data_path)
    for i in os.listdir(data_path):
        if not compare_date(datetime_now,pd.read_parquet(data_path + i)['date'].iloc[-1]):
            os.remove(data_path+i)
    print('DownloadComplete')
if __name__ == '__main__':
    delet(basic_data_save_path)
    download_basic_data_all(basic_data_save_path)

