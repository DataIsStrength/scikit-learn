# -*- coding: utf-8 -*-
"""
Lasso回归和Ridge回归
"""

'''
1.欠拟合与过拟合：
①欠拟合：一个模型在训练集上不能获得好的拟合，并且在测试集上也不能很好地拟合，
此时认为这个模型出现了欠拟合的现象。(模型过于简单)
原因：学习到数据的特征过少。
解决办法：增加数据的特征数量。
②过拟合：一个模型在训练集上能够很好地拟合， 但是在测试集上不能很好地拟合，
此时认为这个模型出现了过拟合的现象。(模型过于复杂)
原因：原始特征过多且存在一些嘈杂特征，模型尝试去兼顾各个测试数据点。
解决办法：减小高次项特征的影响，方法为正则化，即在损失函数中增加惩罚项。
2.正则化：
①l1正则化：惩罚项是所有系数的绝对值之和（l1范数），正则化后的损失函数如下：
J(w)=(1/2m)Σ(h(xi)-yi)^2+λΣ|wj|
其中m为样本量，加入系数1/2m是为了便于求导，λ为正则化力度，用来决定惩罚力度，
使用l2正则化损失函数的回归模型即套索回归（Lasso回归）。
②l2正则化：惩罚项是所有系数的平方和（l2范数），正则化后的损失函数如下：
J(w)=(1/2m)Σ(h(xi)-yi)^2+λΣwj^2
使用l2正则化损失函数的回归模型即岭回归（Ridge回归）。
③对比：l1正则化会导致某些特征的系数为0，相当于直接删除某些特征，
而l2正则化会削弱一些特征的影响，但不会将这些特征删除，
在处理实际问题时l2正则化更加常用。
'''

#导入波士顿房价数据集
from sklearn.datasets import load_boston
#导入数据集划分函数
from sklearn.model_selection import train_test_split
#导入标准化类
from sklearn.preprocessing import StandardScaler
#导入Lasso回归和Ridge回归预估器类
from sklearn.linear_model import Lasso,Ridge
#导入MSE函数
from sklearn.metrics import mean_squared_error
#导入R-square函数
from sklearn.metrics import r2_score

#导入数据
boston=load_boston()
print('数据特征:\n',boston.feature_names)

#划分数据集
x_train,x_test,y_train,y_test=train_test_split(boston.data,boston.target)

'''
Lasso()和Ridge()的参数：
①alpha：正则化力度，默认值为1。
②solver：优化方法，设置为auto时会自动选择合适的优化方法，
比如在数据较大时采用SAG。
③normalize：数据是否标准化，默认值为False,设置为True时可以将输入的数据标准化。
④Lasso()和Ridge()也有一些和SGDRegressor()相同的参数，比如迭代次数max_iter。
'''

#实例化预估器类，正则化力度1，自动选择优化方法，数据标准化，迭代10000次
#estimator=Lasso(alpha=1,normalize=True,max_iter=10000)
estimator=Ridge(alpha=1,solver='auto',normalize=True,max_iter=10000)
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
r2=r2_score(y_test,y_predict)
print('均方误差:\n',error)
print('R-square:\n',r2)