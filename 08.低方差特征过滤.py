# -*- coding: utf-8 -*-
"""
低方差特征过滤
"""

'''
冗余变量是指在对象中不具有明显差异的变量，
删除冗余变量的方法包括：
1.Filter（过滤式）：主要探究特征本身特点、特征与特征和目标值之间关联
①方差选择法：低方差特征过滤
②相关系数法
2.Embedded（嵌入式）：算法自动选择特征（特征与目标值之间的关联）
①决策树：信息熵、信息增益
②正则化：L1、L2
③深度学习：卷积等
'''

import pandas as pd
#导入方差阈值类
from sklearn.feature_selection import VarianceThreshold

#导入数据
data=pd.read_csv('factor_returns.csv')
print(data)

#实例化方差阈值类，设置方差阈值为1
transfer=VarianceThreshold(threshold=1)
#调用fit_transform
data=transfer.fit_transform(data.iloc[:,1:10])
print('删除低方差特征的结果:\n',data)
print('形状:\n',data.shape)