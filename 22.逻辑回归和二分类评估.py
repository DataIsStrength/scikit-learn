# -*- coding: utf-8 -*-
"""
逻辑回归和二分类评估
"""

'''
1.逻辑回归：
一种适用于二分类问题的分类算法，逻辑回归模型的输入值是线性回归的结果，
通过sigmoid函数（激活函数）返回样本属于某个类别的概率，
当sigmoid函数的值达到阈值（通常默认为0.5），将样本划分为正例。
2.sigmoid函数：
g(h(xi))=(1+e^(-h(xi)))^(-1)
值域为(0,1)，激活函数的阈值默认为0.5。
3.对数似然损失：
逻辑回归模型的目标值是离散的，所以不适用最小二乘法的损失函数，
需要引入对数似然损失函数，对于单个样本i，损失函数如下：
if yi=1, cost(h(xi),yi)=-ln(h(xi))
if yi=0, cost(h(xi),yi)=-ln(1-h(xi))
根据该函数，当yi=1时，如果h(xi)=1，则损失为0，如果h(xi)=0，则损失为无穷大；
类似地，当yi=0时，h(xi)=0时损失为0，h(xi)=1时损失无穷大。
对于样本总体，将以上函数加总，可以得到总体的对数似然损失函数：
cost(h(xi),yi)=-Σ(yi*ln(h(xi))+(1-yi)ln(1-h(xi)))
该函数在形式上类似于信息熵公式，常使用梯度下降法进行优化。
'''

import numpy as np
import pandas as pd
#导入数据集划分函数
from sklearn.model_selection import train_test_split
#导入标准化类
from sklearn.preprocessing import StandardScaler
#导入逻辑回归预估器类
from sklearn.linear_model import LogisticRegression
#导入分类评估函数
from sklearn.metrics import classification_report
#导入AUC函数
from sklearn.metrics import roc_auc_score


#设置字段名
column_names=['Sample code number','Clump Thickness','Uniformity of Cell Size',
              'Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size',
              'Bare Nuclei','Bland Chromatin','Normal Nucleoli', 'Mitoses', 'Class']

#导入数据，并添加字段名
data=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
                  names=column_names)

#用nan代替？，并删除缺失值
data=data.replace(to_replace='?',value=np.nan)
data=data.dropna()

#提取特征值
x=data[column_names[1:10]]
#提取目标值
y=data[column_names[10]]

#划分数据集，设置测试集占比30%
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

#特征工程
#标准化
transform=StandardScaler()
x_train=transform.fit_transform(x_train)
x_test=transform.transform(x_test)

'''
LogisticRegression()的参数：
①solver：优化方法。
②penalty：正则化类型。
③C：正则化力度，默认值为1。
'''

#实例化逻辑回归预估器类，优化方法为sag，正则化类型为l2，惩罚力度为1
estimator=LogisticRegression(solver='sag',penalty='l2',C=1)
#训练模型
estimator.fit(x_train,y_train)

#显示回归系数
print('回归系数:',estimator.coef_)
#显示偏置
print('偏置:',estimator.intercept_)

'''
1.混淆矩阵：
                  预测结果
        [    |正例       |反例         ]
真实结果[正例|真正例（TP）|伪反例（FN)  ]
       [反例|伪正例（FP）|真反例（TN） ]
2.精确率和召回率
①精确率：预测结果为正例样本中真实为正例的比例
公式：precision=TP/(TP+FP)
②召回率：真实为正例的样本中预测结果为正例的比例，
公式：recall=TP/(TP+FN)
召回率是一个非常常用的指标，反映了检查的全面性和对正样本的区分能力，
比如在本例中，召回率就是实际上是恶性的肿瘤被预测为恶性的比率。
3.f1-score：精确率和召回率的调和平均数，反映了模型的稳健型，
当f1-score大时，模型的精确率和召回率都大，公式如下：
f1-score=2TP/(2TP+FN+FP)=2precision*recall/(precision+recall)
4.classification_report()的输入值：
①y_true：真实目标值。
②y_pred：估计器预测目标值。
③labels：指定类别对应的数字。
④target_names：目标类别名称。
5.classification_report()的返回值：
每个类别精确率,召回率和f1-score。
6.TPR和FPR：
①TPR：即真阳性率，所有真实类别为1的样本中，预测类别为1的比例，公式如下：
TPR=TP/(TP+FN)
②FPR：即假阳性率，所有真实类别为0的样本中，预测类别为1的比例，公式如下：
FPR=FP/(FP+FN)
7.ROC曲线：
分类器在不同阈值下的FPR-TPR曲线。
8.AUC值：
ROC曲线和FPR轴围成的面积，AUC值的取值范围是(0.5,1)，
AUC值越接近0.5分类器越差，反之，AUC值越接近1则分类器越好，
AUC=1时分类器是完美的，不管设定什么样的阈值都可以得到正确的预测。
'''

#验证模型
print('预测结果:',estimator.predict(x_test))
#评估模型
#准确率
print('预测准确率:',estimator.score(x_test,y_test))
#精确率与召回率
print('精确率和召回率:', classification_report(y_test,estimator.predict(x_test),labels=[2,4],target_names=['良性','恶性']))
#AUC值
print('AUC值:',roc_auc_score(y_test,estimator.predict(x_test)))