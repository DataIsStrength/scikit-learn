# -*- coding: utf-8 -*-
"""
k近邻算法（模型调优）
"""

'''
m折交叉验证：
将训练集划分为m份，每次取1份作为验证集，其余作为训练集来训练模型，
训练模型m次，以m次的平均准确率作为模型的准确率。
超参数网格搜索：
通常情况下，有很多参数是需要手动指定的（如k-近邻算法中的K值），
这样的参数叫作超参数。
对模型预设几种超参数组合，每组超参数都采用交叉验证来进行评估，
最后选出最优参数组合建立模型的过程即超参数网格搜索。
'''

#导入鸢尾花数据集
from sklearn.datasets import load_iris
#导入数据集划分函数
from sklearn.model_selection import train_test_split
#导入标准化类
from sklearn.preprocessing import StandardScaler
#导入knn预估器类
from sklearn.neighbors import KNeighborsClassifier
#导入网格搜索函数
from sklearn.model_selection import GridSearchCV

#获取数据
iris=load_iris()

#划分数据集，设置随机种子
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,random_state=0)

#特征工程
#标准化
transfer=StandardScaler()
x_train=transfer.fit_transform(x_train)
#用和训练集相同的标准化流程将测试集标准化
x_test=transfer.transform(x_test)

#实例化knn预估器类，因为要对模型进行调优，所以不设置k值
estimator=KNeighborsClassifier()

'''
网格搜索函数GridSearchCV()有3个参数：
①estimator：估计器对象。
②param_grid：备选的估计器参数（比如knn算法的k值）。
③cv：指定几折交叉验证。
'''

#设置备选的预估器参数
param_dict={'n_neighbors':[1,3,5,7,9,11]}
#对预估器进行10折交叉验证的网格搜索
estimator=GridSearchCV(estimator,param_grid=param_dict,cv=10)

#训练模型
estimator.fit(x_train,y_train)

#验证模型
y_predict=estimator.predict(x_test)
print('预测结果:\n',y_predict)

#模型评估
#比对真实值和预测值
print('对比预测值和真实值:\n',y_test==y_predict)
#调用转换器类的score函数计算准确率
score=estimator.score(x_test,y_test)
print('准确率:\n',score)
