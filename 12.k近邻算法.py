# -*- coding: utf-8 -*-
"""
k近邻算法
"""

'''
1.knn算法定义：
如果一个样本在特征空间中的k个最相似(即特征空间中最邻近)
的样本中的大多数属于某一个类别，则该样本也属于这个类别。
衡量邻近程度的距离可以是特征空间中的欧式距离，也可以是更加广义的Minkovski距离。
k的取值不宜过大过小，
当k取值过小时，易受极端值影响，
当k取值过大时，易受样本不均衡的影响。
在对数据进行k近邻学习之前，首先需要对特征进行无量纲化处理，比如标准化。
2.knn算法流程：
①获取数据
②数据划分：训练集和测试集
③特征工程：标准化
④训练knn分类器
⑤模型评估
'''

#导入鸢尾花数据集
from sklearn.datasets import load_iris
#导入数据集划分函数
from sklearn.model_selection import train_test_split
#导入标准化类
from sklearn.preprocessing import StandardScaler
#导入knn预估器类
from sklearn.neighbors import KNeighborsClassifier


#获取数据
iris=load_iris()

#划分数据集，设置随机种子
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,random_state=0)

'''
sklearn的组件主要分为转换器类和预估器类两大类，
转换器类用于特征工程，预估器类用于机器学习算法，
其中转换器类的函数包括fit(),transform()和fit_transform()，
分别用来计算，转换和计算再转换。
类似地，预估器类也有fit()函数，predict()函数和score()，
分别用来训练模型，预测结果和评估预测精度。
'''

#特征工程
#标准化
transfer=StandardScaler()
x_train=transfer.fit_transform(x_train)
#用和训练集相同的标准化流程将测试集标准化
x_test=transfer.transform(x_test)

#实例化knn预估器类，设置k为3（默认值为5）
estimator=KNeighborsClassifier(n_neighbors=3)
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

'''
k近邻算法总结：
1.优点：
简单，易于理解，易于实现，无需训练。
2.缺点：
懒惰算法，对测试样本分类时的计算量大，内存开销大。
必须指定K值，K值选择不当则分类精度不能保证。
3.使用场景：
小数据场景，几千到几万样本，具体场景具体业务去测试。
'''



