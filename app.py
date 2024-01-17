from flask import Flask,render_template
import pandas as pd
import os
file_path1='C:\\Users\\13756\\OneDrive - integrate collaborative models\\量化记录\\重新建模\\'
file_path2='C:\\Users\\13756\\OneDrive - integrate collaborative models\\量化记录\\重新建模2\\'
save_path='./result/'
app = Flask(__name__)
def result_translate(file_name):
    df1=pd.read_csv(file_name)[:10]
    df1['score1'] = df1['score1'].round(3)
    df1['score2'] = df1['score2'].round(3)
    res1=df1.to_html(index=None)
    for i in range(len(df1)):
        url=f'https://m.10jqka.com.cn/stockpage/33_'+df1['stock'].iloc[i][1:]+'/?client_userid=xXiPr&share_hxapp=gsc&share_action=fenshi.more.share.hyperlink&back_source=hyperlink'
        res1=res1.replace('<td>'+df1['stock'].iloc[i]+'</td>',f'<td><a href="{url}" target="_blank">'+df1['stock'].iloc[i]+'</a></td>')
    return res1
@app.route('/<file_name>')
def hello_world_sample(file_name):
    res=result_translate(save_path+file_name)
    return render_template('index.html', res1=res,file_name=file_name)
@app.route('/')
def hello_world():
    stock_list=os.listdir(save_path)
    res=result_translate(save_path+stock_list[-1])
    return render_template('index.html', res1=res,file_name=stock_list[-1])
@app.route('/result')
def result():
    stock_list=os.listdir(save_path)
    stock_list.reverse()
    return render_template('result.html', result=stock_list)
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=4321)