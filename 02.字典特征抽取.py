# -*- coding: utf-8 -*-
"""
字典特征抽取
"""

#从sklearn的特征抽取模块调用向量化类
from sklearn.feature_extraction import DictVectorizer

#创建数据集（字典的列表）
data = [{'city': '北京','temperature':100}, 
        {'city': '上海','temperature':60}, 
        {'city': '深圳','temperature':30}]
#实例化向量化类
transfer=DictVectorizer(sparse=False)
'''
sparse参数的默认值是True，当不设置sparse参数时返回值是sparse矩阵，
sparse矩阵将非零值和对应的位置表示出来，
比如
[  0.   1.   0. 100.]
[  1.   0.   0.  60.]
[  0.   0.   1.  30.]
的sparse矩阵是
(0, 1)    1.0
(0, 3)    100.0
(1, 0)    1.0
(1, 3)    60.0
(2, 2)    1.0
(2, 3)    30.0
sparse矩阵只包含非零值，所以可以节省存储空间。
'''
#调用向量化类的fit_transform函数
data=transfer.fit_transform(data)
print('返回的结果:\n', data)
#打印特征名字
print('特征名字:\n', transfer.get_feature_names())
'''
字典特征抽取的应用场景：
1.当数据集的特征比较多是，将数据集的特征转化为字典类型，并进行字典特征抽取。
2.当数据本身就是字典类型时。
'''
