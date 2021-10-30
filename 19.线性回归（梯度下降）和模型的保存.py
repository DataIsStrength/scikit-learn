# -*- coding: utf-8 -*-
"""
线性回归（梯度下降）和模型的保存
"""

'''
1.回归：
目标值连续的机器学习。
2.线性回归：
对特征值和目标值之间的关系进行拟合的一种分析模型。
在单个特征值和目标值形成的二维空间中向一个直线拟合，
在两个特征值和目标值形成的三位空间中向一个平面拟合，
更多特征值的情况依此类推。
线性关系的线性模型：h(w)=w1*x1+w2*x2+w3*x3+...+wn*xn+b=w.T*x+b
其中w=[w1,w2,w3,...,wn]，x=[x1,x2,x3,...,xn]，b为偏置（截距）。
广义线性模型：h(w)=w1*x1+w2*x2^2+w3*x3^3+...+wn*xn^n+b=w.T*x+b
该模型描述目标值与特征值之间的非线性关系，
但是目标值与系数矩阵w=[w1,w2,w3,...,wn]的关系仍然是线性的。
3.损失函数：
对于m个样本的回归模型，定义损失函数为：
J(w)=(h(x1)-y1)^2+(h(x2)-y2)^2+...+(h(xm)-ym)^2=Σ(h(xi)-yi)^2
损失函数越小则模型的拟合越准确，所以损失函数的优化即min J(w)。
4.优化算法：
通过使损失最小化来使拟合精度最大化的方法即最小二乘法，
解决最小二乘问题有两种常用的优化算法：
①梯度下降：
由某一个系数矩阵开始向损失函数极小值处迭代，对于一些非线性的回归模型，
可能会出现下降至极小值但不是最小值的情况。
②正规方程：w=((X*X.T)^(-1))*X.T*y
其中X是特征值矩阵，y是目标值矩阵，使用正规方程可以直接求出最优结果，
由于正规方程涉及到矩阵的运算，所以只适合处理较小的数据（样本量小于10万）。
'''

#导入波士顿房价数据集
from sklearn.datasets import load_boston
#导入数据集划分函数
from sklearn.model_selection import train_test_split
#导入标准化类
from sklearn.preprocessing import StandardScaler
#导入梯度下降预估器类
from sklearn.linear_model import SGDRegressor
#导入MSE函数
from sklearn.metrics import mean_squared_error
#导入保存和加载模块
import joblib

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

'''
SGDRegressor()的参数：
①loss：损失函数类型，默认值为squared_loss，即普通最小二乘法。
②fit_intercept：是否计算偏置，默认值为True。
③learning_rate：学习率，决定迭代的步长变化，默认值为invscaling,
即步长逐渐减小，如果设置为constant，则步长固定为初始步长eta0。
④eta0：初始步长，默认值为0.01。
⑤max_iter：迭代次数。
⑥penalty：正则化类型，如设置该参数为l1，则使用l1正则化损失函数，
即套索回归（Lasso），设置为l2即岭回归（Ridge）。
'''

#实例化预估器类，设置步长不变，初始步长0.01，迭代次数10000次，正则化类型l1
estimator=SGDRegressor(learning_rate='constant',eta0=0.01,max_iter=10000,penalty='l1')
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

'''
1.均方误差：
反映预测值和真实值之间偏离的程度，其公式为：
MSE=(1/m)Σ(yi_pred-yi_true)^2
均方误差越小，则回归模型的预测越精确。
'''

#以pkl文件的形式保存模型
joblib.dump(estimator,'test.pkl')