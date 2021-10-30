# -*- coding: utf-8 -*-
"""
主成分分析
"""

#导入PCA类
from sklearn.decomposition import PCA

data = [[2,8,4,5],
        [6,3,0,8],
        [5,4,9,1]]

'''
PCA类的n_components参数，取值为小数时，表示保留信息的百分比，
取值为整数时，表示下降至指定的维度。
'''

#实例化PCA类，设置保留90%的信息
transfer=PCA(n_components=0.9)
#调用fit_transform
data1=transfer.fit_transform(data)
print('保留90%信息的降维结果为:\n',data1)

#实例化PCA类, 设置下降至3维
transfer2=PCA(n_components=3)
#调用fit_transform
data2=transfer2.fit_transform(data)
print('降维至3维的结果:\n',data2)
