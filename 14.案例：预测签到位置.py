# -*- coding: utf-8 -*-
"""
案例：预测签到位置
"""

import pandas as pd
#导入数据集划分函数
from sklearn.model_selection import train_test_split
#导入标准化类
from sklearn.preprocessing import StandardScaler
#导入knn预估器类
from sklearn.neighbors import KNeighborsClassifier
#导入网格搜索函数
from sklearn.model_selection import GridSearchCV

#处理数据及特征工程
#导入数据
data=pd.read_csv('FBlocation/train.csv')

#数据筛选
data=data.query('x>1.0&x<1.25&y>2.5&y<2.75')

#删除time列
data=data.drop(['time'],axis=1)
print(data)

#删除入住次数少于三次位置
place_count=data.groupby('place_id').count()
tf=place_count[place_count.row_id>3].reset_index()
data=data[data['place_id'].isin(tf.place_id)]


#取出目标值
y=data['place_id']
#取出特征值
x=data.drop(['place_id','row_id'], axis=1)

#划分数据集，设置测试集占比为30%
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

#实例化标准化类
std=StandardScaler()

#将训练集标准化
x_train=std.fit_transform(x_train)
#将测试集标准化
x_test=std.fit_transform(x_test)

#实例化knn预估器类
knn=KNeighborsClassifier()
#设置备选的预估器参数
param_dict={'n_neighbors':[1,3,5,10,20,100,200]}
#对预估器进行10折交叉验证的网格搜索
knn=GridSearchCV(knn,param_grid=param_dict,cv=10)

#训练模型
knn.fit(x_train,y_train)

#验证模型
y_predict=knn.predict(x_test)

#模型评估
print('预测测试集类别:',y_predict)
print('准确率为:',knn.score(x_test,y_test))

#实际预测
#处理样本数据
data1=pd.read_csv('FBlocation/test.csv')
data1=data1.query('x>1.0&x<1.25&y>2.5&y<2.75')
data1=data1.drop(['time'],axis=1)
x1=data1.drop(['row_id'], axis=1)
#调用knn分类器
y_predict1=knn.predict(x1)
print('实际预测结果:',y_predict1)