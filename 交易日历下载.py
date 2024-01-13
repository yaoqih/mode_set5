import requests
import json
import pandas as pd
import os
for i in range(1,13):
    if i<10:
        r = requests.get('http://www.szse.cn/api/report/exchange/onepersistenthour/monthList?month=2024-%s'%"0"+str(i)).text  # 最基本的不带参数的get请求
    else:
        r = requests.get('http://www.szse.cn/api/report/exchange/onepersistenthour/monthList?month=2024-%s'%str(i)).text  # 最基本的不带参数的get请求
    r=json.loads(r)
    df=pd.DataFrame(r['data'])
    if 'trade_calendar.csv' not in os.listdir():
        df.to_csv('trade_calendar.csv',mode='a',index=None,encoding='utf-8')
    else:
        df.to_csv('trade_calendar.csv',mode='a',index=None,header=None,encoding='utf-8')
