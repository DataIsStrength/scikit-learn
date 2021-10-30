# -*- coding: utf-8 -*-
"""
数据预处理-归一化
"""

import pandas as pd
#导入归一化类
from sklearn.preprocessing import MinMaxScaler

"""
归一化的思想是将特征值向一个指定的区间映射，
归一化公式：
X'=(X-min)/(max-min)
X"=X'*(mx-mi)+mi
其中mx和mi分别为指定区间的上下限。
"""

#导入csv数据文件
data=pd.read_csv('dating.csv')
print(data)
#实例化归一化类，指定区间为(2,3)
transfer=MinMaxScaler(feature_range=(2, 3))
#调用fit_transform，选择特征值列
data=transfer.fit_transform(data[['milage','Liters','Consumtime']])
print('最小值最大值归一化处理的结果:\n', data)
