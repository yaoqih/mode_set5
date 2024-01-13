import os
from stock_data_tools import compare_date
import pandas as pd
import numpy as np
from multiprocessing import Pool,Lock
import traceback
from config import basic_data_save_path,other_file_path,参数化数据,mode_data,周期转化后
import joblib
cycle=2
date_remain=100
drop_columns=['CDLHOMINGPIGEON_-2s','VAR_-1s','VAR_-2s','LINEARREG_SLOPE_-1s','LINEARREG_SLOPE_-2s','LINEARREG_ANGLE_-1s','LINEARREG_ANGLE_-2s','BETA_-1s','BETA_-2s','STOCHRSIfastd_-1s','STOCHRSIfastd_-2s','STOCHslowk_-1s','STOCHslowk_-2s','HT_PHASORquadrature_-1s','HT_PHASORquadrature_-2s','ADOSC_-1s','ADOSC_-2s','ULTOSC_-1s','ULTOSC_-2s','PPO_-1s','PPO_-2s','MFI_-1s','MFI_-2s','MACDEXTmacdhist_-1s','MACDEXTmacdhist_-2s','MACDEXTmacdsignal_-1s','MACDEXTmacdsignal_-2s','MACDEXTmacd_-1s','MACDEXTmacd_-2s','CCI_-1s','CCI_-2s','APO_-1s','APO_-2s','CDLTRISTAR_-1s','CDLTRISTAR_-2s','CDLUNIQUE3RIVER_-1s','CDLUNIQUE3RIVER_-2s','CDLUPSIDEGAP2CROWS_-1s','CDLUPSIDEGAP2CROWS_-2s','CDLTASUKIGAP_-1s','CDLTASUKIGAP_-2s','CDLSTICKSANDWICH_-1s','CDLSTICKSANDWICH_-2s','CDLRISEFALL3METHODS_-1s','CDLRISEFALL3METHODS_-2s','CDLSEPARATINGLINES_-1s','CDLSEPARATINGLINES_-2s','CDLMORNINGSTAR_-2s','CDLONNECK_-1s','CDLONNECK_-2s','CDLMATCHINGLOW_-2s','CDLMATHOLD_-1s','CDLMATHOLD_-2s','CDLMORNINGDOJISTAR_-1s','CDLMORNINGDOJISTAR_-2s','CDLKICKING_-1s','CDLKICKING_-2s','CDLKICKINGBYLENGTH_-1s','CDLKICKINGBYLENGTH_-2s','CDLLADDERBOTTOM_-1s','CDLLADDERBOTTOM_-2s','CDLINNECK_-2s','CDLIDENTICAL3CROWS_-1s','CDLIDENTICAL3CROWS_-2s','CDLHIKKAKEMOD_-1s','CDLHIKKAKEMOD_-2s','CDLGAPSIDESIDEWHITE_-1s','CDLGAPSIDESIDEWHITE_-2s','CDLEVENINGDOJISTAR_-1s','CDLCONCEALBABYSWALL_-1s','CDLCONCEALBABYSWALL_-2s','CDLCOUNTERATTACK_-1s','CDLCOUNTERATTACK_-2s','CDLBREAKAWAY_-1s','CDLBREAKAWAY_-2s','CDL3STARSINSOUTH_-1s','CDL3STARSINSOUTH_-2s','CDL3WHITESOLDIERS_-1s','CDL3WHITESOLDIERS_-2s','CDLABANDONEDBABY_-1s','CDLABANDONEDBABY_-2s','CDL3LINESTRIKE_-1s','CDL3LINESTRIKE_-2s','CDL2CROWS_-1s','CDL2CROWS_-2s','CDL3BLACKCROWS_-1s','CDL3BLACKCROWS_-2s','MOM','CDLUPSIDEGAP2CROWS','CDLTRISTAR','CDLTASUKIGAP','CDLSTICKSANDWICH','CDLRISEFALL3METHODS','CDLSEPARATINGLINES','CDLONNECK','CDLMATHOLD','CDLMORNINGDOJISTAR','CDLKICKING','CDLKICKINGBYLENGTH','CDLLADDERBOTTOM','CDLIDENTICAL3CROWS','CDLHIKKAKEMOD','CDLGAPSIDESIDEWHITE','CDLEVENINGDOJISTAR','CDLCONCEALBABYSWALL','CDLCOUNTERATTACK','CDLBREAKAWAY','CDL3STARSINSOUTH','CDL3WHITESOLDIERS','CDLABANDONEDBABY','CDL3LINESTRIKE','CDL2CROWS','CDL3BLACKCROWS','CDL3INSIDE','CDLEVENINGSTAR','CDLINNECK','CDLSTALLEDPATTERN','CDLUNIQUE3RIVER','CDL3INSIDE_-1s','CDLADVANCEBLOCK_-2s','CDLDRAGONFLYDOJI_-2s','CDLEVENINGDOJISTAR_-2s','CDLEVENINGSTAR_-1s','CDLEVENINGSTAR_-2s','CDLGRAVESTONEDOJI_-1s','CDLGRAVESTONEDOJI_-2s','CDLMATCHINGLOW_-1s','CDLMORNINGSTAR_-1s','CDLPIERCING_-1s','CDLPIERCING_-2s','CDLSHOOTINGSTAR_-1s','CDLSHOOTINGSTAR_-2s','CDLSTALLEDPATTERN_-1s','CDLSTALLEDPATTERN_-2s','CDLTAKURI_-1s','STOCHslowd_-1s','STOCHslowd_-2s','STOCHFfastk_-1s','STOCHFfastk_-2s']
drop_columns=[]
def init(l):
	global lock
	lock = l
def 采样_all(path,goal_path,参数):
    '''
    :path:源文件所在路径
    :goal_path:目标文件所放路径
    :name:保存的文件名
    :date_remain:保留的用于回测的天数
    :阈值:分类的阈值
    :manage:1采训练2.采测试0.全部采
    '''
    name=参数.split(',')[0]
    阈值=参数.split(',')[1]
    date_remain=参数.split(',')[2]
    manage=int(参数.split(',')[3])
    codelist = os.listdir(path)
    for i in range(len(codelist)):
        codelist[i]=path+codelist[i]+","+goal_path+','+name+','+str(date_remain)+','+str(阈值)
    # print(codelist[0])
    # exit(0)
    if manage==0 or manage==1:
        if name+','+阈值+','+date_remain+'.csv'not in os.listdir(goal_path):
            采样2(codelist[0])
            lock = Lock()
            pool = Pool(16, initializer=init, initargs=(lock,))
            pool.map_async(采样,codelist[1:])
            pool.close()
            pool.join()
    if manage==2 or manage==0:
        if name+','+阈值+','+date_remain+'测试.csv'not in os.listdir(goal_path):
            测试采样2(codelist[0])
            # for i in codelist[1:]:
            #     测试采样(i)
            lock = Lock()
            pool = Pool(16, initializer=init, initargs=(lock,))
            pool.map_async(测试采样,codelist[1:])
            pool.close()
            pool.join()
def 采样方法(df,阈值,date_next):
    df['target']=-1
    # df['target']=(df['open'].shift(-2)-df['open'].shift(-1))/df['open'].shift(-1)
    df['value']=(df['open'].shift(-2)-df['open'].shift(-1))/df['open'].shift(-1)
    df['next']=(df['open'].shift(-1)-df['close'])/df['close']
    
    df['open'][df['open']==0]=(df['high'][df['open']==0]+df['low'][df['open']==0])/2
    df['open'][df['open']==0]=-0.01

    df['close'][df['close']==0]=(df['high'][df['close']==0]+df['low'][df['close']==0])/2
    df['close'][df['close']==0]=-0.01

    df.loc[df['value']>=阈值, 'target'] = 1
    df.loc[df['value']<阈值, 'target'] = 0
    df.loc[df['value']>0.2, 'target'] = 1
    df.loc[df['value']>0.2, 'value'] = 0.2
    df.loc[df['value']<-0.2, 'target'] = 0
    df.loc[df['value']<-0.2, 'value'] = -0.2

    # df['']
    # df['target']=df['复权pct_chg下']
    df.loc[df['next']>=0.095, 'target'] = 0
    df.drop(['next'],inplace=True,axis=1)
    # df.loc[df[date_next]<阈值, 'target'] = 0
    df.drop(df[df.target ==-1].index,inplace=True,axis=0)
    return df
def 采样(massage):
    '''
    没表头
    '''
    origin_data=massage.split(',')[0]
    goal_path=massage.split(',')[1]+massage.split(',')[2]
    date_remain=int(massage.split(',')[3])
    阈值=float(massage.split(',')[4])
    df=pd.read_csv(origin_data[:-12]+'SZ000001.csv',encoding='utf-8',index_col=0)
    df.drop(drop_columns,axis=1,inplace=True)
    df=df['2010':]
    # df=joblib.load(origin_data[:-12]+'SZ000001.csv')
    date=df.index[-date_remain]
    try:
        # df=joblib.load(origin_data)
        df=pd.read_csv(origin_data,encoding='utf-8',index_col=0)
        df.drop(drop_columns,axis=1,inplace=True)
        df=df['2010':]
        df=df[~(df['close'].isin([0.0]))]
        df=df[~(df['close'].isin([0]))]
        if len(df)>60 and len(df)>date_remain and compare_date(df.index[-date_remain],date) :
            df=采样方法(df,阈值,massage.split(',')[2])
            df=df[~(df['close'].isin([0.0]))]
            df=df[~(df['close'].isin([0]))]
            df=df.iloc[:-date_remain,:]
            df['股票代码']="'"+origin_data.split('\\')[-1][2:-4]
            lock.acquire()
            df.to_csv(goal_path+','+str(阈值)+','+str(date_remain)+'.csv',encoding='utf-8',mode='a',header=None)
            lock.release()
    except:
        traceback.print_exc()
        exit(1)
        # print(massage)
def 采样2(massage):
    '''
    有表头
    '''
    origin_data=massage.split(',')[0]
    goal_path=massage.split(',')[1]+massage.split(',')[2]
    date_remain=int(massage.split(',')[3])
    阈值=float(massage.split(',')[4])
    df=pd.read_csv(origin_data[:-12]+'SZ000001.csv',encoding='utf-8',index_col=0)
    df.drop(drop_columns,axis=1,inplace=True)
    df=df['2010':]
    # df=joblib.load(origin_data[:-12]+'SZ000001.csv')
    date=df.index[-date_remain]
    try:
        # df=joblib.load(origin_data)
        df=pd.read_csv(origin_data,encoding='utf-8',index_col=0)
        df.drop(drop_columns,axis=1,inplace=True)
        df=df['2010':]
        if len(df)>60 and len(df)>date_remain and compare_date(df.index[-date_remain],date) :
            df=采样方法(df,阈值,massage.split(',')[2])
            df=df.iloc[:-date_remain,:]
            df['股票代码']="'"+origin_data.split('\\')[-1][2:-4]
            df.to_csv(goal_path+','+str(阈值)+','+str(date_remain)+'.csv',encoding='utf-8',mode='a')
    except:
        print(massage)
def 测试采样(massage):
    '''
    测试数据
    没表头
    '''
    origin_data=massage.split(',')[0]
    goal_path=massage.split(',')[1]+massage.split(',')[2]
    date_remain=int(massage.split(',')[3])
    阈值=float(massage.split(',')[4])
    df=pd.read_csv(origin_data[:-12]+'SZ000001.csv',encoding='utf-8',index_col=0)
    df.drop(drop_columns,axis=1,inplace=True)
    df=df['2010':]
    # df=joblib.load(origin_data[:-12]+'SZ000001.csv')
    date=df.index[-1]
    try:
        # df=joblib.load(origin_data)
        df=pd.read_csv(origin_data,encoding='utf-8',index_col=0)
        df.drop(drop_columns,axis=1,inplace=True)
        df=df['2010':]
        if len(df)>60 and len(df)>date_remain and compare_date(df.index[-1],date) :
            df=采样方法(df,阈值,massage.split(',')[2])
            df=df.iloc[-date_remain:-3,:]
            df['股票代码']="'"+origin_data.split('\\')[-1][2:-4]
            lock.acquire()
            df.to_csv(goal_path+','+str(阈值)+','+str(date_remain)+'测试'+'.csv',encoding='utf-8',mode='a',header=None)
            lock.release()
    except:
        traceback.print_exc()
        exit(1)
        print(massage)
def 测试采样2(massage):
    '''
    测试数据
    有表头
    '''
    origin_data=massage.split(',')[0]
    goal_path=massage.split(',')[1]+massage.split(',')[2]
    date_remain=int(massage.split(',')[3])
    阈值=float(massage.split(',')[4])
    df=pd.read_csv(origin_data[:-12]+'SZ000001.csv',encoding='utf-8',index_col=0)
    df.drop(drop_columns,axis=1,inplace=True)
    df=df['2010':]
    # df=joblib.load(origin_data[:-12]+'SZ000001.csv')
    date=df.index[-1]
    try:
        # df=joblib.load(origin_data)
        df=pd.read_csv(origin_data,encoding='utf-8',index_col=0)
        df.drop(drop_columns,axis=1,inplace=True)
        df=df['2010':]
        if len(df)>60 and len(df)>date_remain and compare_date(df.index[-1],date) :
            df=采样方法(df,阈值,massage.split(',')[2])
            df=df.iloc[-date_remain:-3,:]
            df['股票代码']="'"+origin_data.split('\\')[-1][2:-4]
            df.to_csv(goal_path+','+str(阈值)+','+str(date_remain)+'测试'+'.csv',encoding='utf-8',mode='a')
    except:
        traceback.print_exc()
        exit(1)
        print(massage)
if __name__ == '__main__':
    # download_basic_data_all(origin_data)
    # 数据周期转化_all(origin_data,2,cycle_result)
    # 参数生成_all(cycle_result,参数化数据_file)
    采样_all(参数化数据,mode_data,'2day,0.03,300,0') 