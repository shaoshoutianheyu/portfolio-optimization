# portfolio-optimization
组合优化（工具：python，tushare）
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 07:24:18 2016

@author: Administrator
"""
# 导入包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco
import tushare as ts

# 股票代码
symbols = ['002337','603636','600848','002736','002604']
noa = len(symbols)

# 利用tushare 获取交易日的日历
ts.set_token('503daef7194d6abcf54c0e96f5641af1407aacc7f738220aebff6d5cbe1a20a4')
mt = ts.Master()
df_date = mt.TradeCal(exchangeCD='XSHG', beginDate='20150101', endDate='20160816', field='calendarDate,isOpen,prevTradeDate')
Date = df_date[df_date['isOpen']==1]["calendarDate"]
# 保留日期为八位字符串格式
Date = Date.apply(lambda x :x.replace('-',''))

# 拉下数据
data = pd.DataFrame({},index=Date)  
for sym in symbols:
    tmp = ts.get_h_data(sym, autype='hfq', start='2015-01-01', end='2016-08-16')['close'] #后复权
    tmp = pd.DataFrame(tmp)
    tmp = tmp.reset_index()
    # 从tushare 拉下的数据也要进行处理 将日期型index转化为八位字符串格式  
    tmp['date'] = tmp['date'].apply(lambda x : x.strftime("%Y%m%d"))
    tmp = tmp.set_index('date')
    data[sym] = tmp['close'] 
 
 
(data / data.ix[0]*100).plot(figsize=(8,5))
#%%
# 收益率序列
rets = np.log(data / data.shift(1))

 
# 蒙特卡洛模拟权重
prets = []
pvols = []
for p in range(2500):
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    prets.append(np.sum(rets.mean()*weights)*252)
    pvols.append(np.sqrt(np.dot(weights.T,np.dot(rets.cov()*252,weights))))
prets = np.array(prets)
pvols = np.array(pvols)


plt.figure(figsize=(8,4))
plt.scatter(pvols,prets,c=prets / pvols,marker = "o")
plt.grid(True)
plt.xlabel("expecter volatility")
plt.ylabel('expected return')
plt.colorbar(label="Sharpe ratio")
#%%


# 传入权重，输出收益率、风险、夏普的一个函数
def statistics(weights):
    weights = np.array(weights)
    pret = np.sum(rets.mean()*weights)*252
    pvol = np.sqrt(np.dot(weights.T,np.dot(rets.cov()*252,weights)))
    return np.array([pret,pvol,pret/pvol])
    
# 最大化夏普指数
def min_func_sharpe(weights):
    return -statistics(weights)[2]
    
    
cons = ({"type":'eq',"fun":lambda x :np.sum(x) - 1})
bnds = tuple((0,1) for x in range(noa))

opts = sco.minimize(min_func_sharpe, noa*[1. / noa,], method="SLSQP", bounds=bnds, constraints=cons)
statistics(opts['x']).round(3) 

# 最小化投资组合的方差
def min_func_variance(weights):
    return statistics(weights)[1]**2
    
optv = sco.minimize(min_func_variance, noa*[1. / noa,], method="SLSQP", bounds=bnds, constraints=cons)
statistics(optv['x']).round(3)


# 关于约束条件的一些说明,其他的一些约束条件根据这个修改即可
cons = ({"type":'eq',"fun":lambda x :np.sum(x) - 1}, #这个约束条件表明权重加起来为1
        { "type":'eq',"fun":lambda x :x[0]-0.05},    # 这个表明第一个权重要大于0.05    x[0] >= 0.05
        {"type":'eq',"fun":lambda x :-x[2]+0.4},     # 这个表明第三个权重要小于等于0.4   x[2] <= 0.4
        )



 


 
 

 

    
    
    
    
    
