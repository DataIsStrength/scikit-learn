# -*- coding: utf-8 -*-
"""
线性回归（正规方程）
"""

#导入波士顿房价数据集
from sklearn.datasets import load_boston
#导入数据集划分函数
from sklearn.model_selection import train_test_split
#导入标准化类
from sklearn.preprocessing import StandardScaler
#导入梯度下降预估器类
from sklearn.linear_model import LinearRegression
#导入MSE函数
from sklearn.metrics import mean_squared_error


#导入数据
boston=load_boston()
print('数据特征:\n',boston.feature_names)

#划分数据集，设置随机种子
x_train,x_test,y_train,y_test=train_test_split(boston.data,boston.target,random_state=0)

#特征工程
#实例化标准化类
transfer=StandardScaler()
x_train=transfer.fit_transform(x_train)
x_test=transfer.transform(x_test)

#实例化正规方程预估器类
estimator=LinearRegression()
#训练模型
estimator.fit(x_train,y_train)

#显示系数
print('回归方程系数为:\n',estimator.coef_)
#显示偏置
print('回归方程偏置为:\n',estimator.intercept_)

#验证模型
y_predict=estimator.predict(x_test)
print('预测结果:\n',y_predict)
#模型评估（均方误差MSE）
error=mean_squared_error(y_test,y_predict)
print('均方误差:\n',error)