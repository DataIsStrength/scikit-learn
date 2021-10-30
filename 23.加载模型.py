# -*- coding: utf-8 -*-
"""
加载模型
"""

#导入波士顿房价数据集
from sklearn.datasets import load_boston
#导入标准化类
from sklearn.preprocessing import StandardScaler
#导入joblib模块
import joblib

#导入数据
boston=load_boston()

#特征工程
#实例化标准化类
transfer=StandardScaler()
x=transfer.fit_transform(boston.data)

#加载模型
estimator=joblib.load('test.pkl')

#预测结果
y_predict=estimator.predict(x)
print('预测结果:',y_predict)

