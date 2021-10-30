# -*- coding: utf-8 -*-
"""
k均值算法
"""

'''
1.无监督学习：
没有目标值的机器学习被称为无监督学习，
常见的无监督学习算法包括主成分分析和k均值算法。
2.k均值算法：
一种迭代求解的聚类分析算法，步骤如下：
①在特征空间中随机取k个点作为初始的聚类中心。
②计算其他每个点到k个中心的距离，并选择最近的一个聚类中心点作为标记类别。
③完成向k个标记中心的聚类之后，重新计算出每个聚类的新中心点，
对于有n个特征x1,x2,...,xn的一组样本，其中心点为：
(mean(x1),mean(x2),...,mean(xn))
④如果计算得出的新中心点与标记的原中心点一样，那么结束聚类，
否则返回到第二步重新进行迭代。
'''

import pandas as pd
#导入PCA类
from sklearn.decomposition import PCA
#导入k均值预估器类
from sklearn.cluster import KMeans
#导入轮廓系数函数
from sklearn.metrics import silhouette_score

#导入数据
products=pd.read_csv('instacart/products.csv')
order_products=pd.read_csv('instacart/order_products__prior.csv')
orders=pd.read_csv('instacart/orders.csv')
aisles=pd.read_csv('instacart/aisles.csv')

#合并表，将user_id和aisle放在一张表上
#用order_id字段合并orders和order_products
tab1=pd.merge(orders,order_products,on=['order_id','order_id'])
#用product_id字段合并tab1和products
tab2=pd.merge(tab1,products,on=['product_id','product_id'])
#用aisle_id字段合并tab2和aisles
tab3=pd.merge(tab2,aisles,on=['aisle_id','aisle_id'])

#用user_id，aisle字段生成交叉表
table=pd.crosstab(tab3['user_id'],tab3['aisle'])
print(table)

#实例化PCA类，保留95%的信息
transfer=PCA(n_components=0.95)
#调用fit_transform
data=transfer.fit_transform(table)
print('降维后的特征数量:',data.shape[1])

#实例化k均值预估器类，设置k值为4
estimator=KMeans(n_clusters=4)

#训练模型
estimator.fit(data)

#预测结果
pre=estimator.predict(data)
print('预测结果:',pre)

'''
1.聚类的评价标准：
一个好的聚类要求同时具有高的内聚度和分离度，
也就是族群内部距离小而族群之间距离大。
2.轮廓系数：
对于单个样本i,其轮廓系数为：
SCi=(bi-ai)/max(ai,bi)
其中ai是i到自身族群内其他样本的平均距离，bi是ai到其他族群样本的距离的最小值。
对于有m个样本的总体，有平均轮廓系数：
SC=(1/m)ΣSCi
SC的取值范围为[-1,1],越接近1则聚类越好，反之越接近-1则聚类越差。
'''

#评估模型
#计算平均轮廓系数
print('平均轮廓系数:',silhouette_score(data,pre))