# -*- coding: utf-8 -*-
"""
计算相关系数
"""

import pandas as pd
import matplotlib.pyplot as plt
#导入pearson相关系数函数
from scipy.stats import pearsonr

data=pd.read_csv("factor_returns.csv")

#选取列名
factor=data.columns[1:-2]

#遍历选取的特征，两两之间计算相关系数
for i in range(len(factor)):
    for j in range(i,len(factor)-1):
        #选择返回值的第一位即相关系数作为输出结果
        print("指标{:s}与指标{:s}之间的相关性大小为{:f}" 
              .format(factor[i],factor[j+1],pearsonr(data[factor[i]],data[factor[j+1]])[0]))
'''
pearsonr()函数有两个返回值，第一个是相关系数，
第二个是相关系数的p值，p值越小则相关系数越显著，
如果只需要相关系数，则需要对返回值进行索引。
'''

#画出相关系数最大的两项的散点图
plt.figure(figsize=(10, 6),dpi=100)
plt.scatter(data['revenue'],data['total_expense'])
plt.show()