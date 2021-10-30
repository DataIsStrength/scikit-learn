# -*- coding: utf-8 -*-
"""
数据预处理-标准化
"""

import pandas as pd
#导入标准化类
from sklearn.preprocessing import StandardScaler

"""
标准化公式：
X'=(X-mean)/σ
相比归一化方法，标准化不易受异常值影响，鲁棒性较好。
"""

#导入csv数据文件
data=pd.read_csv('dating.csv')
print(data)
#实例化标准化类
transfer=StandardScaler()
#调用fit_transform，选择特征值列
data=transfer.fit_transform(data[['milage','Liters','Consumtime']])
print('标准化的结果:\n', data)
#利用标准化类的函数计算均值和方差
print('每一列特征的平均值:\n', transfer.mean_)
print('每一列特征的方差:\n', transfer.var_)
