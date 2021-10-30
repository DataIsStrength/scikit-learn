# -*- coding: utf-8 -*-
"""
案例：泰坦尼克号乘客生存预测
"""

import pandas as pd
#导入字典向量化类
from sklearn.feature_extraction import DictVectorizer
#导入数据集划分函数
from sklearn.model_selection import train_test_split
#导入决策树预估器类
from sklearn.tree import DecisionTreeClassifier

#从网站下载数据
data=pd.read_csv('titanic.csv')

#选择特征值
x=data[['pclass','age','sex']].copy()
#选择目标值
y=data['survived'].copy()


#缺失值处理，inplace设置为True表示对原始数据进行修改
#如果inplace设置为False，则修改后需要赋值给一个新的变量，而原数据不变
x['age'].fillna(x['age'].mean(),inplace=True)

#特征工程
#将x转换成字典数据x.to_dict，设置orient参数可以调整格式，一般常用records
x=x.to_dict(orient='records')
#实例化字典向量化类
transform=DictVectorizer(sparse=False)
#调用fit_transform
x=transform.fit_transform(x)

print(transform.get_feature_names())
print(x)

#划分数据集，设置测试集占比30%
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

#实例化决策树预估器类，设置依据为信息增益最大，树的最大深度为5
estimator=DecisionTreeClassifier(criterion='entropy',max_depth=5)

#训练模型
estimator.fit(x_train,y_train)

#验证和评估模型
print('预测的准确率为:',estimator.score(x_test,y_test))