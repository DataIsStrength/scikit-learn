# -*- coding: utf-8 -*-
"""
导入并划分iris数据集
"""

#调用数据集
from sklearn.datasets import load_iris
#调用数据集划分函数
from sklearn.model_selection import train_test_split

#获取鸢尾花数据集
iris=load_iris()
print('鸢尾花数据集的返回值:\n', iris)

'''
返回值的类型是继承自字典的Bunch,Bunch有5个键值对,
5个键是data,target,feature_names,target_names和DESCR。
'''
#对Bunch的键值对进行索引
print('鸢尾花的特征值:\n',iris['data'])
print('鸢尾花的目标值:\n',iris['target'])
print('鸢尾花特征的名字:\n',iris['feature_names'])
print('鸢尾花目标值的名字:\n',iris['target_names'])
print('鸢尾花的描述:\n',iris['DESCR'])
#也可以用以下方式索引
print('鸢尾花的特征值:\n', iris.data)
print('鸢尾花的目标值:\n', iris.target)
print('鸢尾花特征的名字:\n', iris.feature_names)
print('鸢尾花目标值的名字:\n', iris.target_names)
print('鸢尾花的描述:\n', iris.DESCR)

#划分数据集
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.2,random_state=0)
'''
test_size表示测试集的规模，默认为0.25。
random_state是随机数种子，设置random_state可以使重复运行得到同样的结果。
'''
print('训练集的特征值:\n',x_train)
print('训练集的目标值:\n',y_train)
print('测试集的特征值:\n',x_test)
print('测试集的目标值:\n',y_test)