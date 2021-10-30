# -*- coding: utf-8 -*-
"""
随机森林算法
"""

'''
1.随机森林：
随机森林是包含多个决策树的分类器，其输出的类别是由个别树输出的类别的众数而定。
例如，训练了5个树，其中4个树结果是True，1个数结果False，那么最终结果就是True。
2.Bootstrap抽样：
即随机有放回抽样，是一种重抽样的方法，为了形成随机森林的多个决策树，
要采用Bootstrap抽样，具体过程如下：
①抽取样本：在N个样本中Bootstrap抽取N个，形成一个树的训练数据集。
②选择特征：如果一共有M个特征，则选择m个来训练决策树，m<<M，这样的好处是可以降维。
'''
import pandas as pd
#导入字典向量化类
from sklearn.feature_extraction import DictVectorizer
#导入数据集划分函数
from sklearn.model_selection import train_test_split
#导入随机森林预估器类
from sklearn.ensemble import RandomForestClassifier
#导入网格搜索函数
from sklearn.model_selection import GridSearchCV

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

#实例化随机森林预估器类
estimator=RandomForestClassifier()
#设置备选超参数，n_estimators是决策树的数量，max_depth是单个树的最大深度
param={'n_estimators':[120,200,300,500,800,1200],'max_depth':[5,8,15,25,30]}

#对模型进行2折交叉验证的网格搜索
estimator=GridSearchCV(estimator,param_grid=param,cv=2)
#训练模型
estimator.fit(x_train,y_train)

#验证和评估模型
print('预测的准确率为:',estimator.score(x_test,y_test))

'''
随机森林算法总结：
1.在当前所有算法中，具有极好的准确率。
2.能够有效地运行在大数据集上，处理具有高维特征的输入样本，而且不需要降维。
3.能够评估各个特征在分类问题上的重要性。
'''