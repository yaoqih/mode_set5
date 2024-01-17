import pandas as pd
import os
import datetime
import smtplib  #加载smtplib模块
from email.mime.text import MIMEText
from email.utils import formataddr
from email.mime.multipart import MIMEMultipart
from email.header import Header
import datetime
from email.mime.application import MIMEApplication
from urllib.request import urlopen
from multiprocessing import Pool,Lock
import talib
import shutil
import numpy as np
from datetime import timedelta
import time
import stockstats
import talib
from urllib.request import Request
import psutil
import warnings
import traceback

warnings.filterwarnings("ignore")
averagelist=[15,16,17,21,22,23,24,25,26,28,30,32,33,34,35,36,40,42,43,44,52,53,55,56,57]
averagelist=[15,16,17,21,22,23,24,25,26,28,30,32,33,34,35,36,40,42,43,44,52,53,55,56,57,58,63,61,67]
changeratelist=['换手率','volume','成交金额','总市值','流通市值']
data_path='C:\\stock_data2\\'
other_file_path='C:\\data_test\\other_file\\'
basic_data_save_path='C:\\原始数据\\'
# cpu信息
def get_cpu_info():
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_info = "CPU使用率：%i%%" % cpu_percent
    return cpu_percent
# 内存信息
def get_memory_info():
    virtual_memory = psutil.virtual_memory()
    used_memory = virtual_memory.used/1024/1024/1024
    free_memory = virtual_memory.free/1024/1024/1024
    memory_percent = virtual_memory.percent
    memory_info = "内存使用：%0.2fG，使用率%0.1f%%，剩余内存：%0.2fG" % (used_memory, memory_percent, free_memory)
    return memory_percent
def code_transport(code, method=1):
    """
    代码转换
    1.去"'"
    2.SZ000001
    3.sz000001
    4.0.000001 1.600000(东方财富)
    :code: 股票代码
    :method: 1:返回000001 2.返回SZ000001   3.返回sz000001  4.0.000001 1.600000(东方财富) 4.000001.SZ 600000.SH(tushare)
    """
    if len(code)==7 or code[0]=="'":
        code=code[1:]
    if len(code)==12 and code[-4]=='.':
        code=code[2:-4]
    if len(code)==8 and code[1]=='.':
        code=code[2:]
    if len(code)==8 and(code[:2]=="SZ" or code[:2]=="SH"or code[:2]=="sz"or code[:2]=="sh"):
        code=code[2:]
    if method==1:
        return code
    if method==2:
        if code[0]=="6":
            code="SH"+code
        else:
            code="SZ"+code
        return code
    if method==3:
        if code[0]=="6":
            code="sh"+code
        else:
            code="sz"+code
        return code
    if method==4:
        if code[0]=="6":
            code="1."+code
        else:
            code="0."+code
        return code
    if method==5:
        if code[0]=="6":
            code=code+'.SH'
        else:
            code=code+'.SZ'
        return code
    return code
def getPSY(priceData, period):
    difference = priceData[1:] - priceData[:-1]
    difference = np.append(0, difference)
    difference_dir = np.where(difference > 0, 1, 0)
    psy = np.zeros((len(priceData),))
    psy[:period] *= np.nan
    for i in range(period, len(priceData)):
        psy[i] = (difference_dir[i-period+1:i+1].sum()) / period
    return psy
def DPO(close):
    p = talib.MA(close, timeperiod=11)
    p.shift()
    return close-p
def VHF(close):
    LCP = talib.MIN(close, timeperiod=28)
    HCP = talib.MAX(close, timeperiod=28)
    NUM = HCP - LCP
    pre = close.copy()
    pre = pre.shift()
    DEN = abs(close-close.shift())
    DEN = talib.MA(DEN, timeperiod=28)*28
    return NUM.div(DEN)
def RVI(df):
    close = df.close
    open = df.open
    high = df.high
    low = df.low
    X = close-open+2*(close.shift()-open.shift())+2*(close.shift(periods=2)-open.shift(periods=2))*(close.shift(periods=3)-open.shift(periods=3))/6
    Y = high-low+2*(high.shift()-low.shift())+2*(high.shift(periods=2)-low.shift(periods=2))*(high.shift(periods=3)-low.shift(periods=3))/6
    Z = talib.MA(X, timeperiod=10)*10
    D = talib.MA(Y, timeperiod=10)*10
    return Z/D
def calculateEMA(period, closeArray, emaArray=[]):
    length = len(closeArray)
    nanCounter = np.count_nonzero(np.isnan(closeArray))
    if not emaArray:
        emaArray.extend(np.tile([np.nan], (nanCounter + period - 1)))
        firstema = np.mean(closeArray[nanCounter:nanCounter + period - 1])
        emaArray.append(firstema)
        for i in range(nanCounter + period, length):
            ema = (2 * list(closeArray)[i] + (period - 1) * emaArray[-1]) / (period + 1)
            emaArray.append(ema)
    return np.array(emaArray)
def calculateMACD(closeArray, shortPeriod=12, longPeriod=26, signalPeriod=9):
    ema12 = calculateEMA(shortPeriod, closeArray, [])
    ema26 = calculateEMA(longPeriod, closeArray, [])
    diff = ema12 - ema26

    dea = calculateEMA(signalPeriod, diff, [])
    macd = (diff - dea)

    fast_values = diff
    slow_values = dea
    diff_values = macd
    return fast_values, slow_values, diff_values
def RSI(array_list, periods=14):
    array_list=list(array_list)
    length = len(array_list)
    rsies = [np.nan] * length
    if length <= periods:
        return rsies
    up_avg = 0
    down_avg = 0
    first_t = array_list[:periods + 1]
    for i in range(1, len(first_t)):
        if first_t[i] >= first_t[i - 1]:
            up_avg += first_t[i] - first_t[i - 1]
        else:
            down_avg += first_t[i - 1] - first_t[i]
    up_avg = up_avg / periods
    down_avg = down_avg / periods
    if down_avg==0:
        rsies[periods] =100
    else:
        rs = up_avg / down_avg
        rsies[periods] = 100 - 100 / (1 + rs)
    for j in range(periods + 1, length):
        up = 0
        down = 0
        if array_list[j] >= array_list[j - 1]:
            up = array_list[j] - array_list[j - 1]
            down = 0
        else:
            up = 0
            down = array_list[j - 1] - array_list[j]
        up_avg = (up_avg * (periods - 1) + up) / periods
        down_avg = (down_avg * (periods - 1) + down) / periods
        if down_avg==0:
            rsies[j]=100
        else:
            rs = up_avg / down_avg
            rsies[j] = 100 - 100 / (1 + rs)
    return rsies
def last_train_data(initel=0):
    '''
    以今日为基准上一个交易日
    :initel:负为下一个，正为上一个
    '''
    trandcal=pd.read_csv(other_file_path+'交易日历.csv',encoding='gb18030')
    #提取所有开盘日
    tradingdays = trandcal[trandcal['jybz'] == 1]   # 开盘日
    if int(time.strftime("%H", time.localtime()))<9:
        nowTime=time.mktime(time.strptime(str(getYesterday()),"%Y-%m-%d"))
    else:
        nowTime = time.mktime(time.strptime(datetime.datetime.now().strftime('%Y-%m-%d'),"%Y-%m-%d"))
    if '-' in str(trandcal['jyrq'].values[-1]):
        trandtime = time.mktime(time.strptime(str(trandcal['jyrq'].values[-1]),"%Y-%m-%d"))
    elif '/' in str(trandcal['jyrq'].values[-1]):
        trandtime = time.mktime(time.strptime(str(trandcal['jyrq'].values[-1]),"%Y/%m/%d"))
    else:
        trandtime = time.mktime(time.strptime(str(trandcal['jyrq'].values[-1]),"%Y%m%d"))
    i=1
    while int(nowTime)-int(trandtime)<0:
        i+=1
        if '-' in str(tradingdays['jyrq'].values[-i]):
            trandtime = time.mktime(time.strptime(str(tradingdays['jyrq'].values[-i]),"%Y-%m-%d"))
        elif '/' in str(tradingdays['jyrq'].values[-i]):
            trandtime = time.mktime(time.strptime(str(tradingdays['jyrq'].values[-i]),"%Y/%m/%d"))
        else:
            trandtime = time.mktime(time.strptime(str(tradingdays['jyrq'].values[-i]),"%Y%m%d"))
    trandtime=str(tradingdays['jyrq'].values[-i-initel])
    if '/' not in trandtime and '-' not in trandtime:
        trandtime=trandtime[:4]+'-'+trandtime[4:6]+'-'+trandtime[6:]
    return trandtime
def init(l):
	global lock
	lock = l
def getYesterday():
    '''
    以今日为基准获取昨天日期
    '''
    yesterday = datetime.date.today() + datetime.timedelta(-1)
    return yesterday
def compare_date(time1, time2):
    '''
    比较时间是否相同
    支持20010830，2001-08-30，2001/08/30
    :initel:负为下一个，正为上一个
    '''
    if '/'in time1:
        time1 = time1.replace('/','-')
    if '/'in time2:
        time2 = time2.replace('/','-')
    d1 = datetime.datetime.strptime(time1, '%Y-%m-%d')
    d2 = datetime.datetime.strptime(time2, '%Y-%m-%d')
    if d1==d2:
        return True
    else:
        return False
def 回测(df,dayafter,columns_list,rename=False):
    '''
    :df:回测的DataFrame
    :dayafter:回测df几天后的数据
    :columns_list:需要取回的列
    :rename:将取回的列重命名list
    :return:添加列后的DataFrame
    '''
    for k in columns_list:
        df[k]=''
    for i in range(len(df)):
        date=df['交易日期'].iat[i]
        code=df['股票代码'].iat[i]
        code=code_transport(code,2)
        if code+'.csv' in os.listdir(data_path):
            df1=pd.read_csv(data_path+code+'.csv',encoding='gb18030')
            index=-1
            for j in range(len(df1)-1,-1,-1):
                if compare_date(df1['交易日期'].iat[j],date):
                    index=j
                    break
            index+=dayafter
            if index+1>len(df1):
                print('没有%s %s %s日后的信息' % (code,date,dayafter))
                df.drop(columns_list,axis=1,inplace=True)
                return df
            for k in columns_list:
                df[k].iat[i]=df1[k].iat[index]
        else:
            print(code+'not found')
    for i in columns_list:
        df.drop(df[df[i]==""].index,inplace=True)
    if rename:
        df.rename(columns=dict(zip(columns_list,rename)),inplace=True)
    return df
def sent_email(code,receiver='1375626371@qq.com',filename=None,default="注意事项：\n1.此列表仅供参考，存在一定的风险，请使用者根据自己的情况自行选择，本人不对列表的准确性负责。\n2.此列表只对文件内个股交易日期的下一个交易日有效,下一个交易日之后的涨跌幅未测试，准确度未知\n3.此列表只针对数据分析，暂时无消息面的分析，使用前请自行检查该公司在上个交易日结束后至下一个交易日是否有负面消息。\n4.本列表目前在实验阶段，准确率还有波动。\n5.ST股票风险相对较高,请提高警惕,做好风险防控\n6.入市有风险，投资需谨慎。\n如果使用该列表视作仔细阅读，理解并同意上述说明，自行承担风险。"):
    '''
    :code:邮件标题
    :receiver:收件人地址可以是list
    :filename:邮件附件路径可以是list
    :default:默认字段
    '''
    #带附件的邮件发送
    _user = "1375626371@qq.com"
    _pwd = "dwvwwkupcmpmjjjb"
    _to = receiver

    # 如名字所示Multipart就是分多个部分
    msg = MIMEMultipart()
    msg["Subject"] = code+'('+str(datetime.date.today()) +")"
    msg["From"] = _user
    msg["To"] = _to

    # ---这是文字部分---
    #part = MIMEText("生如夏花之绚烂，死如秋叶之静谧")
    part = MIMEText(default)
    msg.attach(part)

    # ---这是附件部分---
    # txt类型附件（同文件夹）
    if filename:
        if isinstance(filename,list):
            for i in filename:
                if filename[0] == 'c':
                    前缀='报告'
                else:
                    前缀=''
                if filename == 'c.csv'or filename == 'b.csv'or filename == 'd.csv':
                    df=pd.read_csv(filename,encoding='gb18030')
                    date=df['交易日期'].iat[-1]
                else:
                    date=datetime.date.today()
                part = MIMEApplication(open(i, 'rb').read())
                part.add_header('Content-Disposition', 'attachment', filename=str(date)+'.csv')
                msg.attach(part)
        else:
            if filename[0] == 'c':
                前缀='报告'
            else:
                前缀=''
            if filename == 'c.csv'or filename == 'b.csv'or filename == 'd.csv':
                df=pd.read_csv(filename,encoding='gb18030')
                date=df['交易日期'].iat[-1]
            else:
                date=datetime.date.today()
            part = MIMEApplication(open(filename, 'rb').read())
            part.add_header('Content-Disposition', 'attachment', filename=前缀+str(date)+'.csv')
            msg.attach(part)
    s = smtplib.SMTP("smtp.qq.com",25, timeout=30)  # 连接smtp邮件服务器,端口默认是25
    s.login(_user, _pwd)  # 登陆服务器
    s.sendmail(_user, _to, msg.as_string())  # 发送邮件
    s.close()
def download_basic_data(inputfile):
    try:
        trandtime=last_train_data()
        inputfile=str(inputfile)
        if len(inputfile)<6:
            inputfile='0'*(6-len(inputfile))+inputfile
        if inputfile[0]=='6':
            inputfile='SH'+inputfile+'.csv'
        else:
            inputfile='SZ'+inputfile+'.csv'
        if inputfile not in os.listdir(basic_data_save_path):
            code=''
            url='http://quotes.money.163.com/service/chddata.html?code=%s&start=19910101&end=20300101'
            if inputfile[2]=='6':
                code='0'+inputfile[2:8]
            elif inputfile[2]=='0' or inputfile[2]=='3':
                code='1'+inputfile[2:8]
            content=urlopen(url%(code)).read()
            if len(content)==0:
                content=urlopen(url%(code)).read()
            if len(content)==0:
                content=urlopen(url%(code)).read()
            if len(content)==0:
                print('HTTP,Error '+inputfile)
                return -1
            content=content.decode('gb18030','ignore').replace('\n','')
            # lock.acquire()
            f = open(basic_data_save_path+inputfile,'w')
            f.write(content)
            f.close()
            time.sleep(0.01)
            df=pd.read_csv(basic_data_save_path+inputfile,encoding='gb18030')
            # lock.release()
            df=df[~(df['涨跌幅'].isin([None]))]
            df=df[~(df['换手率'].isin([None]))]
            df=df[~(df['换手率'].isin(['None']))]
            df=df[~(df['开盘价'].isin([0.0]))]
            df=df[~(df['开盘价'].isin([0]))]
            df=df[~(df['名称'].isin(['']))]
            df['换手率']=pd.DataFrame(df['换手率'],dtype=np.float)
            df.rename(columns={'日期':'交易日期','前收盘':'前收盘价'},inplace=True)
            df.sort_values(by=['交易日期'], inplace=True)
            df.drop(['成交笔数'],axis=1,inplace=True)
            df.to_csv(basic_data_save_path+inputfile,encoding='gb18030',index=False)
            if df.shape[0]>0 and len(df)>50 and compare_date(df['交易日期'].iloc[-1],datetime)and compare_date(df['交易日期'].iloc[-1],traintime):
                time.sleep(0.01)
            else:
                if inputfile in os.listdir(basic_data_save_path):
                    os.remove(basic_data_save_path+inputfile)
    except:
        if inputfile in os.listdir(basic_data_save_path):
            os.remove(basic_data_save_path+inputfile)
        print(inputfile)
        print('error')
def download_data(inputfile):
    try:
        trandtime=last_train_data()
        inputfile=str(inputfile)
        if len(inputfile)<6:
            inputfile='0'*(6-len(inputfile))+inputfile
        if inputfile[0]=='6':
            inputfile='SH'+inputfile+'.csv'
        else:
            inputfile='SZ'+inputfile+'.csv'
        if inputfile not in os.listdir(data_path):
            code=''
            url='http://quotes.money.163.com/service/chddata.html?code=%s&start=19910101&end=20300101'
            if inputfile[2]=='6':
                code='0'+inputfile[2:8]
            elif inputfile[2]=='0' or inputfile[2]=='3':
                code='1'+inputfile[2:8]
            content=urlopen(url%(code)).read()
            if len(content)==0:
                content=urlopen(url%(code)).read()
            if len(content)==0:
                content=urlopen(url%(code)).read()
            if len(content)==0:
                print('HTTP,Error '+inputfile)
                return -1
            content=content.decode('gb18030','ignore').replace('\n','')
            lock.acquire()
            f = open(data_path+inputfile,'w')
            f.write(content)
            f.close()
            time.sleep(0.01)
            df=pd.read_csv(data_path+inputfile,encoding='gb18030')
            lock.release()
            df=df[~(df['涨跌幅'].isin([None]))]
            df=df[~(df['开盘价'].isin([0.0]))]
            df=df[~(df['开盘价'].isin([0]))]
            df=df[~(df['名称'].isin(['']))]
            df['换手率']=pd.DataFrame(df['换手率'],dtype=np.float)
            if df.shape[0]>0 and len(df)>50 and  (df['日期'].iloc[0] ==trandtime or df['日期'].iloc[-1]==trandtime.replace('-','/')):
                df.rename(columns={'日期':'交易日期','前收盘':'前收盘价'},inplace=True)
                df.sort_values(by=['交易日期'], inplace=True)
                df['交易日期']=pd.to_datetime(df['交易日期'],format=r"%Y/%m/%d")
                df['复权涨跌幅']=df['收盘价']/df['前收盘价']-1
                df['复权涨跌幅'].iloc[0]=df['收盘价'].iloc[0]/df['开盘价'].iloc[0]-1
                df['复权因子'] = (df['复权涨跌幅'] + 1).cumprod()
                initial_price = df.iloc[0]['收盘价'] / (1 + df.iloc[0]['复权涨跌幅'])  # 计算上市价格
                df['收盘价_后复权'] = initial_price * df['复权因子']  # 相乘得到复权价
                df['开盘价_后复权'] = df['开盘价'] / df['收盘价'] * df['收盘价_后复权']
                df['最高价_后复权'] = df['最高价'] / df['收盘价'] * df['收盘价_后复权']
                df['最低价_后复权'] = df['最低价'] / df['收盘价'] * df['收盘价_后复权']
                rename_dict={'开盘价_后复权':'open','收盘价_后复权':'close','最高价_后复权':'high','最低价_后复权':'low','成交量':'volume','成交额（千元）':'amount'}
                df.rename(columns=rename_dict,inplace=True)
                if len(df)>33:
                    df['fast'],df['slow'],df['DIFF']=calculateMACD(df['close'])
                    df['fast']=df['fast'].round(decimals=2)
                    df['slow']=df['slow'].round(decimals=2)
                    df['DIFF']=df['DIFF'].round(decimals=2)
                df['RSI_6']=RSI(df['close'],periods=6)
                df['RSI_12']=RSI(df['close'],periods=12)
                df['RSI_24']=RSI(df['close'],periods=24)
                df['MA_5'] = df['close'].rolling(5,min_periods=1).mean()
                df['MA_10'] = df['close'].rolling(10,min_periods=1).mean()
                df['MA_20'] = df['close'].rolling(20,min_periods=1).mean()
                df['MA_30'] = df['close'].rolling(30,min_periods=1).mean()
                df['MA_60'] = df['close'].rolling(60,min_periods=1).mean()
                df['MA_6']=df['close'].rolling(6,min_periods=1).mean()
                df['BIAS_6']=(df['close']-df['MA_6'])/df['MA_6']*100
                df['MA_12']=df['close'].rolling(12,min_periods=1).mean()
                df['BIAS_12']=(df['close']-df['MA_12'])/df['MA_12']*100
                df['MA_24']=df['close'].rolling(24,min_periods=1).mean()
                df['BIAS_24']=(df['close']-df['MA_24'])/df['MA_24']*100
                df['volume_min']=df['volume'].rolling(20,min_periods=1).min()
                stock = stockstats.StockDataFrame.retype(df)
                stock.get('change')
                stock.get('rate')
                stock.get('middle')
                stock.get('boll')
                stock.get('kdjk')
                stock.get('cr')
                stock.get('cci')
                stock.get('tr')
                stock.get('atr')
                stock.get('um')
                stock.get('pdi')
                stock.get('trix')
                stock.get('tema')
                stock.get('vr')
                stock.get('dma')
                stock['wr_10']=stock['wr_10']
                stock['wr_6']=stock['wr_6']
                stock['MOM'] = talib.MOM(stock['close'], timeperiod=5)
                stock['OBV'] = talib.OBV(stock['close'], stock['volume'])
                stock['CMO'] = talib.CMO(stock['close'], timeperiod=10)
                stock['ROC'] = talib.ROC(stock['close'], timeperiod=10)
                stock['ROC_ma10'] = talib.MA(stock['ROC'] , timeperiod=10)
                stock['PSY']=getPSY(stock['close'].values,5)
                stock['PSY_ma10'] = talib.MA(stock['PSY'] , timeperiod=10)
                stock['OBV_ma10'] = talib.MA(stock['OBV'] , timeperiod=10)
                stock['DPO']=DPO(stock['close'])
                stock['VHF']=VHF(stock['close'])
                stock['RVI']=RVI(stock)
                stock['VHF_ma10'] = talib.MA(stock['VHF'] , timeperiod=10)
                stock['RVI_ma10'] = talib.MA(stock['RVI'] , timeperiod=10)
                stock=stock[['交易日期','股票代码','名称','收盘价','涨跌幅','换手率','volume','成交金额','总市值','流通市值','复权涨跌幅','close','open','high','low','fast','slow','diff','rsi_6','rsi_12','rsi_24','ma_5','ma_10','ma_20','ma_30','ma_60','ma_6','bias_6','ma_12','bias_12','ma_24','bias_24','close_20_sma','close_20_mstd','boll','boll_ub','boll_lb','kdjk_9','kdjd_9','kdjj_9','middle_14_sma','cci','close_-1_s','tr','atr','high_delta','low_delta','pdi_14','mdi_14','dx_14','dx_6_ema','adx_6_ema','trix','tema','vr','close_10_sma','close_50_sma','dma','VHF_ma10','RVI_ma10','OBV_ma10','DPO','RVI','VHF','wr_6','MOM','OBV','CMO','ROC','ROC_ma10','PSY','PSY_ma10','wr_10']]
                #stock.drop(changeratelist,axis=1,inplace=True)
                stock.replace(float('inf'),0,inplace=True)
                stock.replace(float('-inf'),0,inplace=True)
                for i in range(len(averagelist)):
                    stock[str(list(stock.columns)[averagelist[i]])] =stock[str(list(stock.columns)[averagelist[i]])]/stock['收盘价']  #stock.apply(lambda x: x[str(list(stock.columns)[averagelist[i]-1])] / x['close'], axis=1)
                for i in changeratelist:
                    k=stock[i].shift(1)
                    k.drop([stock.index[0]],inplace=True)
                    stock.drop([stock.index[0]],inplace=True)
                    stock[i+'_rate'] =(stock[i]-k)/k #stock.apply(lambda x: x[str(list(stock.columns)[averagelist[i]-1])] / x['close'], axis=1)
                stock.drop(changeratelist,axis=1,inplace=True)
                stock.replace(float('inf'),0,inplace=True)
                stock.replace(float('-inf'),0,inplace=True)
                for i in range(10,len(stock.columns)-1):
                    stock[list(stock.columns)[i]+'_-1']=stock.iloc[:,i:i+1].shift(1)
                    stock[list(stock.columns)[i]+'_-1s']=stock[list(stock.columns)[i]]-stock[list(stock.columns)[i]+'_-1']
                    # stock[list(stock.columns)[i]+'_-2']=stock.iloc[:,i:i+1].shift(2)
                    # stock[list(stock.columns)[i]+'_-2s']=stock[list(stock.columns)[i]]-stock[list(stock.columns)[i]+'_-2']
                stock=stock.fillna(0)
                stock.replace('None',0,inplace=True)
                # rename_dict={'open':'开盘价_后复权','clsoe':'收盘价_后复权','high':'最高价_后复权','low':'最低价_后复权','volume':'成交量（手）','amount':'成交额（千元）','涨跌幅':'涨跌额','涨跌额':'涨跌幅'}
                # df.rename(columns=rename_dict,inplace=True)
                stock=stock[50:]
                # lock.acquire()
                stock.to_csv(data_path+inputfile,encoding='gb18030',index=None,float_format='%.4f')
                # lock.release()
                return 0
            else:
                if inputfile in os.listdir(data_path):
                    os.remove(data_path+inputfile)
    except:
        if inputfile in os.listdir(data_path):
            os.remove(data_path+inputfile)
        print(inputfile)
        print('error')
def deleteNullFile(path):
    '''删除所有大小为0的文件'''
    files = os.listdir(path)
    for file in files:
        if os.path.getsize(path+file)  < 2000:   #获取文件大小
            os.remove(path+file)
            print(file + " deleted.")
    print('deleteNullFile complete')
def movefile(srcfile,dstfile):
    '''
    移动文件
    从哪里来，到哪里去
    '''
    fpath,fname=os.path.split(dstfile)    #分离文件名和路径
    if not os.path.exists(fpath):
        os.makedirs(fpath)                #创建路径
    shutil.copyfile(srcfile,dstfile)      #复制文件
def copyfile(srcfile,dstfile):
    '''
    复制文件
    从哪里来，到哪里去
    '''
    fpath,fname=os.path.split(dstfile)    #分离文件名和路径
    if not os.path.exists(fpath):
        os.makedirs(fpath)                #创建路径
    shutil.copyfile(srcfile,dstfile)      #复制文件
def get_prise(code):
    '''
    获取现在股票价格
    '''
    try:
        code=code_transport(code,3)
        headers = {
        "User-Agent": "Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)",
        }
        nowpriceurl='https://hq.sinajs.cn/list=%s'%code
        req = Request(nowpriceurl,headers=headers)
        nowprice=urlopen(req).read().decode('GB18030')
        if nowprice[-1]!=';':
            nowprice=urlopen(nowpriceurl).read().decode('GB18030')
        nowprice=dict(zip(['open','before','now','high','low','buy1','volume','turnover'],nowprice.split(',')[1:9]))#  1,2,3,4,5
        if 'dict' not in str(type(nowprice)):
            nowprice=get_prise(code[2:])
        for i in nowprice.keys():
            nowprice[i]=float(nowprice[i])
        return nowprice
    except:
        nowprice=get_prise(code[2:])
def download_basic_data_all(data_path):
    '''
    下载所有原始数据
    :data_path:存放地址
    '''
    if 'SZ000001.csv' not in os.listdir(data_path)  or pd.read_csv(data_path+'SZ000001.csv',encoding='gb18030')['交易日期'].iat[-1]!=last_train_data():
        delet(data_path)
    global basic_data_save_path
    basic_data_save_path = data_path
    df = pd.read_csv(other_file_path+'股票列表.csv',encoding='gb18030').code
    codelist = list(df)
    download_basic_data('1')
    download_basic_data('1')
    datetime_now=pd.read_csv(basic_data_save_path + 'SZ000001.csv', encoding='gb18030')['交易日期'].iloc[-1]
    lock = Lock()
    pool = Pool(16, initializer=init, initargs=(lock,))
    pool.map_async(download_basic_data,codelist)
    pool.close()
    pool.join()
    deleteNullFile(data_path)
    for i in os.listdir(basic_data_save_path):
        if not compare_date(datetime_now,pd.read_csv(basic_data_save_path + i, encoding='gb18030')['交易日期'].iloc[-1]):
            os.remove(basic_data_save_path+i)
    print('DownloadComplete')
def download_data_all():
    if 'SZ000001.csv' not in os.listdir(data_path) or pd.read_csv(data_path+'SZ000001.csv',encoding='gb18030').shape[1]<30 or pd.read_csv(data_path+'SZ000001.csv',encoding='gb18030')['交易日期'].iat[-1]!=last_train_data():
        delet(data_path)
    df = pd.read_csv(other_file_path+'股票列表.csv',encoding='gb18030').code
    codelist = list(df)
    lock = Lock()
    pool = Pool(16, initializer=init, initargs=(lock,))
    pool.map_async(download_data,codelist)
    pool.close()
    pool.join()
    print('DownloadComplete')
def open_sec():
    '''
    开机了多少秒
    '''
    open_time=datetime.datetime.strptime(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(psutil.boot_time())),'%Y-%m-%d %H:%M:%S')
    now = datetime.datetime.now()
    interval = now - open_time
    sec = interval.days*24*3600 + interval.seconds
    return sec
def get_allcode_pricenow(filename):
    '''
    获取股票上一个交易日情况
    :filename:文件路径
    '''
    df=pd.read_csv(filename,encoding='gb18030')
    drop_list=[]
    df['pricenow']=0
    df['现涨跌幅']=0
    df['开盘价']=0
    df['开盘涨跌幅']=0
    df['实际收益']=0
    for i in range(len(df)):
        code=df['股票代码'].iloc[i][1:]
        price=get_prise(code)
        while 'dict'not in str(type(price)):
            price=get_prise(code)
        if price['open']==0:
            drop_list.append(i)
        df['pricenow'].iloc[i]=price['now']
        df['现涨跌幅'].iloc[i]=round((df['pricenow'].iloc[i]- price['before'])/price['before'],4)*100
        df['开盘价'].iloc[i]=price['open']
        df['开盘涨跌幅'].iloc[i]=round((price['open']- price['before'])/price['before'],4)*100
    for i in drop_list:
        df.drop(i,axis=0,inplace=True) 
    df=df.reindex(range(len(df)),method='bfill') 
    df['实际收益']=((df['pricenow']-df['开盘价'])/df['开盘价'])*100
    return df
def get_allcode_pricepast(df,file_path):
    '''
    获取股票上一个交易日情况
    :filename:文件路径
    :file_path:对应数据库存放地址   
    '''
    
    date=df['交易日期'].iloc[0]
    drop_list=[]
    df['pricenow']=0
    df['开盘价']=0
    df['收盘价']=0
    df['最高价']=0
    df['最低价']=0
    df['后开盘价']=0
    df['前收盘价（复权）']=0
    df['开盘涨跌幅']=0
    df['实际收益']=0
    df['第三天开盘收益']=0
    df['第三天收盘收益']=0
    df['第四天收盘收益']=0
    df['第四天开盘收益']=0
    for i in range(len(df)):
        code=df['股票代码'].iloc[i][1:]
        code=code_transport(code,2)
        if code+'.csv' in os.listdir(file_path):
            df1=pd.read_csv(file_path+code+'.csv',encoding='gb18030')
            df1['开盘价'] = pd.to_numeric(df1['开盘价'],errors='coerce')
            df1['收盘价'] = pd.to_numeric(df1['收盘价'],errors='coerce')
            for b in range(len(df1)-1,-1,-1):
                if compare_date(df1['交易日期'].iloc[b],date):
                    break
            df['前收盘价（复权）'].iloc[i]=df1['收盘价'].iloc[b]
            if b+1<len(df1):
                df['开盘价'].iloc[i]=df1['开盘价'].iloc[b+1]
                df['收盘价'].iloc[i]=df1['收盘价'].iloc[b+1]
                df['最高价'].iloc[i]=df1['最高价'].iloc[b+1]
                df['最低价'].iloc[i]=df1['最低价'].iloc[b+1]
                df['开盘涨跌幅'].iloc[i]=round((df1['开盘价'].iloc[b+1]- df1['收盘价'].iloc[b])/df1['收盘价'].iloc[b],4)*100
            if b+2<len(df1):
                df['后开盘价'].iloc[i]=df1['开盘价'].iloc[b+2]
                df['pricenow'].iloc[i]=df1['收盘价'].iloc[b+2]
                df['第三天收盘收益'].iloc[i]=((df1['收盘价'].iloc[b+2]-df1['开盘价'].iloc[b+1])/df1['开盘价'].iloc[b+1])
            if b+3<len(df1):
                df['第四天开盘收益'].iloc[i]=((df1['开盘价'].iloc[b+3]-df1['开盘价'].iloc[b+1])/df1['开盘价'].iloc[b+1])
                df['第四天收盘收益'].iloc[i]=((df1['收盘价'].iloc[b+3]-df1['开盘价'].iloc[b+1])/df1['开盘价'].iloc[b+1])
            df.fillna(inplace=True,value=0.0)
    for i in drop_list:
        df.drop(i,axis=0,inplace=True) 
    df=df.reindex(range(len(df)),method='bfill') 
    df['第三天开盘收益']=((df['后开盘价']-df['开盘价'])/df['开盘价'])
    return df
def delet(roottdir):
    '''
    删除路径中所有文件
    :filename:文件路径
    '''
    filelist=os.listdir(roottdir)                #列出该目录下的所有文件名
    for f in filelist:
        filepath = os.path.join(roottdir, f )   #将文件名映射成绝对路劲
        if os.path.isfile(filepath):            #判断该文件是否为文件或者文件夹
            os.remove(filepath)                 #若为文件，则直接删除
def save_name_deal(date):
    '''
    日期规范化为2001-08-30
    '''
    if  '/' in date:
        date=date.replace('/','-')
    if  '\\' in date:
        date=date.replace('\\','-')
    date_split=date.split('-')
    if len(date.split('-')[1])==1:
        date_split[1]='0'+date_split[1]
    if len(date.split('-')[2])==1:
        date_split[2]='0'+date_split[2]
    return '-'.join(date_split)
def 数据周期转化_all(path,cycle,goal_path):
    '''
    周期线的生成
    :path:数据源的地址
    :cycle:天数
    :goal_path:目标储存位置
    '''
    if 'SZ000001.csv' not in os.listdir(goal_path)  or not compare_date(pd.read_csv(goal_path+'SZ000001.csv',encoding='gb18030')['交易日期'].iat[-1],last_train_data()):
        delet(goal_path)
    codelist = os.listdir(path)
    codelist_all=[]
    for i in range(len(codelist)):
        if codelist[i] not in os.listdir(goal_path):
            codelist_all.append(path+codelist[i]+","+str(cycle)+','+goal_path)
    lock = Lock()
    pool = Pool(16, initializer=init, initargs=(lock,))
    pool.map_async(数据周期转化,codelist_all)
    pool.close()
    pool.join()
    print('周期转化完成')
def 数据周期转化(filename):
    try:
        cycle=int(filename.split(',')[1])
        goal_path=filename.split(',')[2]
        filename=filename.split(',')[0]
        df=pd.read_csv(filename,encoding='gb18030')
        num=len(df)
        m,n = map(int,[len(df.columns),num])
        matrix = [[0]*(m)]*(n)
        df1=pd.DataFrame(matrix,columns=list(df.columns))
        # df1=df1.reindex(columns=list(df.columns))
        df['涨跌额'] = pd.to_numeric(df['涨跌额'],errors='coerce')
        df['换手率'] = pd.to_numeric(df['换手率'],errors='coerce')
        df['成交量'] = pd.to_numeric(df['成交量'],errors='coerce')
        df['成交金额'] = pd.to_numeric(df['成交金额'],errors='coerce')
        df['总市值'] = pd.to_numeric(df['总市值'],errors='coerce')
        df['流通市值'] = pd.to_numeric(df['流通市值'],errors='coerce')
        a=0
        for i in range(len(df)-1,cycle,-1):
            df1['交易日期'].iloc[a]=df['交易日期'].iloc[i]
            df1['收盘价'].iloc[a]=df['收盘价'].iloc[i]
            df1['开盘价'].iloc[a]=df['开盘价'].iloc[i-cycle+1]
            df1['最高价'].iloc[a]=max(list(df['最高价'].iloc[i-cycle+1:i+1]))
            df1['最低价'].iloc[a]=min(list(df['最低价'].iloc[i-cycle+1:i+1]))
            df1['前收盘价'].iloc[a]=df['前收盘价'].iloc[i-cycle+1]
            df1['涨跌额'].iloc[a]=np.sum(list(df['涨跌额'].iloc[i-cycle+1:i+1]))
            df1['换手率'].iloc[a]=np.average((list(df['换手率'].iloc[i-cycle+1:i+1])))
            df1['成交量'].iloc[a]=np.sum((list(df['成交量'].iloc[i-cycle+1:i+1])))
            df1['成交金额'].iloc[a]=np.sum((list(df['成交金额'].iloc[i-cycle+1:i+1])))
            df1['总市值'].iloc[a]=np.average((list(df['总市值'].iloc[i-cycle+1:i+1])))
            df1['流通市值'].iloc[a]=np.average((list(df['流通市值'].iloc[i-cycle+1:i+1])))
            a=a+1
        df1.drop(df1[df1['交易日期']==0].index,inplace=True)
        df1['股票代码']=df['股票代码'].iloc[-1]
        df1['名称']=df['名称'].iloc[-1]
        df1['涨跌幅']=(df1['收盘价']-df1['收盘价'].shift(periods=-1))/df1['收盘价'].shift(periods=-1)
        df1.sort_values(by=['交易日期'], inplace=True)
        df1.to_csv(goal_path+filename.split('\\')[-1],encoding='gb18030',index=None)
    except:
        print(filename)
        traceback.print_exc()
def 参数生成_all(path,goal_path):
    '''
    其他参数
    :path:数据源的地址
    :goal_path:目标储存位置
    '''
    if 'SZ000001.csv' not in os.listdir(goal_path) or pd.read_csv(goal_path+'SZ000001.csv',encoding='gb18030').shape[1]<30 or pd.read_csv(goal_path+'SZ000001.csv',encoding='gb18030')['交易日期'].iat[-1]!=last_train_data():
        delet(goal_path)
    codelist = os.listdir(path)
    for i in range(len(codelist)):
        codelist[i]=path+codelist[i]+','+goal_path
    lock = Lock()
    pool = Pool(16, initializer=init, initargs=(lock,))
    pool.map_async(参数生成,codelist)
    pool.close()
    pool.join()
    print('参数化完成')
def 参数生成(filename):
    goal_path=filename.split(',')[1]
    filename=filename.split(',')[0]
    df=pd.read_csv(filename,encoding='gb18030')
    df.sort_values(by=['交易日期'], inplace=True)
    df['交易日期']=pd.to_datetime(df['交易日期'],format=r"%Y/%m/%d")
    df['复权涨跌幅']=df['收盘价']/df['前收盘价']-1
    df['复权涨跌幅'].iloc[0]=df['收盘价'].iloc[0]/df['开盘价'].iloc[0]-1
    df['复权因子'] = (df['复权涨跌幅'] + 1).cumprod()
    initial_price = df.iloc[0]['收盘价'] / (1 + df.iloc[0]['复权涨跌幅'])  # 计算上市价格
    df['收盘价_后复权'] = initial_price * df['复权因子']  # 相乘得到复权价
    df['开盘价_后复权'] = df['开盘价'] / df['收盘价'] * df['收盘价_后复权']
    df['最高价_后复权'] = df['最高价'] / df['收盘价'] * df['收盘价_后复权']
    df['最低价_后复权'] = df['最低价'] / df['收盘价'] * df['收盘价_后复权']
    rename_dict={'开盘价_后复权':'open','收盘价_后复权':'close','最高价_后复权':'high','最低价_后复权':'low','成交量':'volume','成交额（千元）':'amount'}
    df.rename(columns=rename_dict,inplace=True)
    if len(df)>33:
        df['fast'],df['slow'],df['DIFF']=calculateMACD(df['close'])
        df['fast']=df['fast'].round(decimals=2)
        df['slow']=df['slow'].round(decimals=2)
        df['DIFF']=df['DIFF'].round(decimals=2)
    df['RSI_6']=RSI(df['close'],periods=6)
    df['RSI_12']=RSI(df['close'],periods=12)
    df['RSI_24']=RSI(df['close'],periods=24)
    df['MA_5'] = df['close'].rolling(5,min_periods=1).mean()
    df['MA_10'] = df['close'].rolling(10,min_periods=1).mean()
    df['MA_20'] = df['close'].rolling(20,min_periods=1).mean()
    df['MA_30'] = df['close'].rolling(30,min_periods=1).mean()
    df['MA_60'] = df['close'].rolling(60,min_periods=1).mean()
    df['MA_6']=df['close'].rolling(6,min_periods=1).mean()
    df['BIAS_6']=(df['close']-df['MA_6'])/df['MA_6']*100
    df['MA_12']=df['close'].rolling(12,min_periods=1).mean()
    df['BIAS_12']=(df['close']-df['MA_12'])/df['MA_12']*100
    df['MA_24']=df['close'].rolling(24,min_periods=1).mean()
    df['BIAS_24']=(df['close']-df['MA_24'])/df['MA_24']*100
    df['volume_min']=df['volume'].rolling(20,min_periods=1).min()
    stock = stockstats.StockDataFrame.retype(df)
    stock.get('change')
    stock.get('rate')
    stock.get('middle')
    stock.get('boll')
    stock.get('kdjk')
    stock.get('cr')
    stock.get('cci')
    stock.get('tr')
    stock.get('atr')
    stock.get('um')
    stock.get('pdi')
    stock.get('trix')
    stock.get('tema')
    stock.get('vr')
    stock.get('dma')
    stock['wr_10']=stock['wr_10']
    stock['wr_6']=stock['wr_6']
    stock['MOM'] = talib.MOM(stock['close'], timeperiod=5)
    stock['OBV'] = talib.OBV(stock['close'], stock['volume'])
    stock['CMO'] = talib.CMO(stock['close'], timeperiod=10)
    stock['ROC'] = talib.ROC(stock['close'], timeperiod=10)
    stock['ROC_ma10'] = talib.MA(stock['ROC'] , timeperiod=10)
    stock['PSY']=getPSY(stock['close'].values,5)
    stock['PSY_ma10'] = talib.MA(stock['PSY'] , timeperiod=10)
    stock['OBV_ma10'] = talib.MA(stock['OBV'] , timeperiod=10)
    stock['DPO']=DPO(stock['close'])
    stock['VHF']=VHF(stock['close'])
    stock['RVI']=RVI(stock)
    stock['VHF_ma10'] = talib.MA(stock['VHF'] , timeperiod=10)
    stock['RVI_ma10'] = talib.MA(stock['RVI'] , timeperiod=10)
    stock=stock[['交易日期','股票代码','名称','收盘价','涨跌幅','换手率','volume','成交金额','总市值','流通市值','复权涨跌幅','close','open','high','low','fast','slow','diff','rsi_6','rsi_12','rsi_24','ma_5','ma_10','ma_20','ma_30','ma_60','ma_6','bias_6','ma_12','bias_12','ma_24','bias_24','close_20_sma','close_20_mstd','boll','boll_ub','boll_lb','kdjk_9','kdjd_9','kdjj_9','middle_14_sma','cci','close_-1_s','tr','atr','high_delta','low_delta','pdi_14','mdi_14','dx_14','dx_6_ema','adx_6_ema','trix','tema','vr','close_10_sma','close_50_sma','dma','VHF_ma10','RVI_ma10','OBV_ma10','DPO','RVI','VHF','wr_6','MOM','OBV','CMO','ROC','ROC_ma10','PSY','PSY_ma10','wr_10']]
    #stock.drop(changeratelist,axis=1,inplace=True)
    stock.replace(float('inf'),0,inplace=True)
    stock.replace(float('-inf'),0,inplace=True)
    for i in range(len(averagelist)):
        stock[str(list(stock.columns)[averagelist[i]])] =stock[str(list(stock.columns)[averagelist[i]])]/stock['收盘价']  #stock.apply(lambda x: x[str(list(stock.columns)[averagelist[i]-1])] / x['close'], axis=1)
    for i in changeratelist:
        k=stock[i].shift(1)
        k.drop([stock.index[0]],inplace=True)
        stock.drop([stock.index[0]],inplace=True)
        stock[i+'_rate'] =(stock[i]-k)/k #stock.apply(lambda x: x[str(list(stock.columns)[averagelist[i]-1])] / x['close'], axis=1)
    stock.drop(changeratelist,axis=1,inplace=True)
    stock.replace(float('inf'),0,inplace=True)
    stock.replace(float('-inf'),0,inplace=True)
    for i in range(10,len(stock.columns)-1):
        stock[list(stock.columns)[i]+'_-1']=stock.iloc[:,i:i+1].shift(1)
        stock[list(stock.columns)[i]+'_-1s']=stock[list(stock.columns)[i]]-stock[list(stock.columns)[i]+'_-1']
        # stock[list(stock.columns)[i]+'_-2']=stock.iloc[:,i:i+1].shift(2)
        # stock[list(stock.columns)[i]+'_-2s']=stock[list(stock.columns)[i]]-stock[list(stock.columns)[i]+'_-2']
    stock=stock.fillna(0)
    stock.replace('None',0,inplace=True)
    # rename_dict={'open':'开盘价_后复权','clsoe':'收盘价_后复权','high':'最高价_后复权','low':'最低价_后复权','volume':'成交量（手）','amount':'成交额（千元）','涨跌幅':'涨跌额','涨跌额':'涨跌幅'}
    # df.rename(columns=rename_dict,inplace=True)
    stock=stock[50:]
    # lock.acquire()
    stock.to_csv(goal_path+filename.split('\\')[-1],encoding='gb18030',index=None,float_format='%.4f')
    # lock.release()
    return 0
def 采样_all(path,goal_path,name,date_remain,阈值,manage):
    '''
    :path:源文件所在路径
    :goal_path:目标文件所放路径
    :name:保存的文件名
    :date_remain:保留的用于回测的天数
    :阈值:分类的阈值
    :manage:1采训练2.采测试0.全部采
    '''
    codelist = os.listdir(path)
    for i in range(len(codelist)):
        codelist[i]=path+codelist[i]+","+goal_path+','+name+','+str(date_remain)+','+str(阈值)
    # print(codelist[0])
    # exit(0)
    if manage==0 or manage==1:
        采样2(codelist[0])
        lock = Lock()
        pool = Pool(16, initializer=init, initargs=(lock,))
        pool.map_async(采样,codelist[1:])
        pool.close()
        pool.join()
    if manage==2 or manage==0:
        测试采样2(codelist[0])
        lock = Lock()
        pool = Pool(16, initializer=init, initargs=(lock,))
        pool.map_async(测试采样,codelist[1:])
        pool.close()
        pool.join()
def 采样方法(df,阈值):
    df['复权涨跌幅下']=(df['close'].shift(-2)-df['open'].shift(-1))/df['open'].shift(-1)
    # df['target']=df['复权涨跌幅下']
    df.loc[df['复权涨跌幅下']>=阈值, 'target'] = 1
    df.loc[df['复权涨跌幅下']<阈值, 'target'] = 0
    return df
def 采样(massage):
    '''
    没表头
    '''
    origin_data=massage.split(',')[0]
    goal_path=massage.split(',')[1]+massage.split(',')[2]
    date_remain=int(massage.split(',')[3])
    阈值=float(massage.split(',')[4])
    df=pd.read_csv(origin_data[:-12]+'SZ000001.csv',encoding='gb18030')
    date=df['交易日期'].iloc[-date_remain]
    try:
        df=pd.read_csv(origin_data,encoding='gb18030')
        df=df[~(df['收盘价'].isin([0.0]))]
        df=df[~(df['收盘价'].isin([0]))]
        if len(df)>60 and len(df)>date_remain and compare_date(df['交易日期'].iloc[-date_remain],date) :
            df=采样方法(df,阈值)
            df=df[~(df['收盘价'].isin([0.0]))]
            df=df[~(df['收盘价'].isin([0]))]
            df=df.iloc[3:-date_remain,:]
            lock.acquire()
            df.to_csv(goal_path+str(阈值)+','+str(date_remain)+'.csv',encoding='gb18030',index=False,mode='a',header=None)
            lock.release()
    except:
        print(massage)
def 采样2(massage):
    '''
    有表头
    '''
    origin_data=massage.split(',')[0]
    goal_path=massage.split(',')[1]+massage.split(',')[2]
    date_remain=int(massage.split(',')[3])
    阈值=float(massage.split(',')[4])
    df=pd.read_csv(origin_data[:-12]+'SZ000001.csv',encoding='gb18030')
    date=df['交易日期'].iloc[-date_remain]
    try:
        df=pd.read_csv(origin_data,encoding='gb18030')
        if len(df)>60 and len(df)>date_remain and compare_date(df['交易日期'].iloc[-date_remain],date) :
            df=采样方法(df,阈值)
            df=df.iloc[3:-date_remain,:]
            df.to_csv(goal_path+str(阈值)+','+str(date_remain)+'.csv',encoding='gb18030',index=False,mode='a')
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
    df=pd.read_csv(origin_data[:-12]+'SZ000001.csv',encoding='gb18030')
    date=df['交易日期'].iloc[-1]
    try:
        df=pd.read_csv(origin_data,encoding='gb18030')
        if len(df)>60 and len(df)>date_remain and compare_date(df['交易日期'].iloc[-1],date) :
            df=采样方法(df,阈值)
            df=df.iloc[-date_remain:-2,:]
            lock.acquire()
            df.to_csv(goal_path+str(阈值)+','+str(date_remain)+'测试'+'.csv',encoding='gb18030',index=False,mode='a',header=None)
            lock.release()
    except:
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
    df=pd.read_csv(origin_data[:-12]+'SZ000001.csv',encoding='gb18030')
    date=df['交易日期'].iloc[-1]
    try:
        df=pd.read_csv(origin_data,encoding='gb18030')
        if len(df)>60 and len(df)>date_remain and compare_date(df['交易日期'].iloc[-1],date) :
            df=采样方法(df,阈值)
            df=df.iloc[-date_remain:-2,:]
            df.to_csv(goal_path+str(阈值)+','+str(date_remain)+'测试'+'.csv',encoding='gb18030',index=False,mode='a')
    except:
        print(massage)