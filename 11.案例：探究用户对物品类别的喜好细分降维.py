# -*- coding: utf-8 -*-
"""
案例：探究用户对物品类别的喜好细分降维
"""

import pandas as pd
#导入PCA类
from sklearn.decomposition import PCA

'''
数据：
1.商品信息products.csv：
字段：product_id,product_name,aisle_id,department_id
2.订单与商品信息order_products__prior.csv：
字段：order_id,product_id,add_to_cart_order,reordered 
3.用户的订单信息orders.csv：
字段：order_id,user_id,eval_set,order_number,order_dow,order_hour_of_day,days_since_prior_order 
4.商品所属具体物品类别aisles.csv：
字段：aisle_id,aisle
'''     

#导入数据
products=pd.read_csv('instacart/products.csv')
order_products=pd.read_csv('instacart/order_products__prior.csv')
orders=pd.read_csv('instacart/orders.csv')
aisles=pd.read_csv('instacart/aisles.csv')

'''
pd.merge()函数可以左右合并DataFrame，
on参数可以设置合并所依据的字段。
'''

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
#打印数据表格形状
print(data.shape)
