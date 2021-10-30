# -*- coding: utf-8 -*-
"""
朴素贝叶斯算法
"""

'''
朴素贝叶斯算法原理：
由贝叶斯公式：
P(目标值|特征值)
=P(特征值|目标值)*P(目标值)/P(特征值)
=P(特征值1|目标值)*P(特征值2|目标值)*...*P(特征值n|目标值)*P(目标值)
 /P(特征值1)*P(特征值2)*...*P(特征值n)
计算特征值属于某一特定目标值的概率，然后取概率最大的目标值给特征值分类。
朴素贝叶斯算法的前提条件是特征之间相互独立。
'''

#导入新闻数据集
from sklearn.datasets import fetch_20newsgroups
#导入数据集划分函数
from sklearn.model_selection import train_test_split
#导入tf-idf向量化类
from sklearn.feature_extraction.text import TfidfVectorizer
#导入朴素贝叶斯预估器类
from sklearn.naive_bayes import MultinomialNB


#导入数据
news=fetch_20newsgroups(subset="all")

#划分数据集，设置测试集占比30%
x_train,x_test,y_train,y_test=train_test_split(news.data,news.target,test_size=0.3)

#特征工程
#文本特征抽取tf-idf
transfer=TfidfVectorizer()
x_train=transfer.fit_transform(x_train)
#用训练集的特征转换器抽取测试集的特征
x_test=transfer.transform(x_test)

'''
拉普拉斯平滑系数：
在样本量较小时，单个特征的条件概率可能为0，对结果的准确性造成影响，
为了避免这种情况，要引入拉普拉斯平滑系数。
拉普拉斯平滑系数修正的条件概率：
P(Fi|C)=(Ni+α)/(N+αm)
其中Ni是特征Fi的样本个数，N是条件C的样本个数，m是特征总数，
α是拉普拉斯平滑系数，通常默认为1。
'''

#实例化朴素贝叶斯预估器类，设置拉普拉斯平滑系数为1
estimator=MultinomialNB(alpha=1.0)
#训练朴素贝叶斯分类器
estimator.fit(x_train,y_train)

#验证模型
y_predict=estimator.predict(x_test)
print('y_predict:\n',y_predict)

#模型评估
#比对真实值和预测值
print('直接比对真实值和预测值:\n',y_test==y_predict)
#计算准确率
score=estimator.score(x_test,y_test)
print('准确率为:\n',score)

'''
朴素贝叶斯算法总结：
1.优点：
朴素贝叶斯模型发源于古典数学理论，有稳定的分类效率。
对缺失数据不太敏感，算法也比较简单，常用于文本分类。
分类准确度高，速度快。
2.缺点：
由于使用了特征相互独立的假设，所以如果特征相关时其效果不好。
'''


