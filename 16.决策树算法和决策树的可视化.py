# -*- coding: utf-8 -*-
"""
决策树算法和决策树的可视化
"""

'''
1.信息：
信息奠基人香农（Shannon）认为“信息是用来消除随机不确定性的东西”，
这一定义被人们看作是经典性定义并加以引用。
2.信息量：
随机事件发生的概率越小，则消除其不确定性需要的信息越多，也就是信息量越大，
反之概率越大则信息量越小，一定发生的事件信息量为0，
那么有事件xi的信息量公式：
h(xi)=-logP(xi)
该公式的对数底数是可以随意选择的，一般习惯用2作为底数。
3.信息熵：
随机变量X的信息熵是在结果产生之前对可能产生的信息量的期望，公式如下：
H(X)=-ΣP(xi)*logP(xi)
信息熵可以作为一个系统复杂程度的度量，如果系统越复杂，
出现不同情况的种类越多，则系统的信息熵越大。
4.信息增益：
特征A可以降低数据集D的信息熵，那么信息熵的降低为信息增益g(D,A),
定义为集合D的信息熵H(D)与在特征A的条件下D的信息条件熵H(D|A)之差，
其公式为：
g(D,A)=H(D)-H(D|A)
其中H(D)为D的信息熵，H(D|A)为D在A条件下的信息条件熵。
5.以信息增益划分的决策树：
将所有特征按照信息增益的大小排序，先判断信息增益大的特征，
后判断信息增益小的特征，形成决策树。
决策树还存在其他的划分依据，比如信息增益比最大，基尼系数最小等。
'''

#导入鸢尾花数据集
from sklearn.datasets import load_iris
#导入数据集划分函数
from sklearn.model_selection import train_test_split
#导入决策树分类器和决策树可视化函数
from sklearn.tree import DecisionTreeClassifier,export_graphviz


#导入数据
iris=load_iris()

#划分数据集并设置随机种子
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,random_state=0)

#实例化决策树预估器类
#设置criterion参数可以确定决策树的依据，默认值为gini，即基尼系数最小，
#此处设置为信息增益
estimator=DecisionTreeClassifier(criterion='entropy')

#训练模型
estimator.fit(x_train,y_train)

#验证模型
y_predict=estimator.predict(x_test)
print('y_predict:\n', y_predict)

#模型评估
#比对真实值和预测值
print('直接比对真实值和预测值:\n', y_test==y_predict)
#计算准确率
score=estimator.score(x_test,y_test)
print('准确率为:\n',score)

#可视化决策树
export_graphviz(estimator,out_file='iris_tree.dot',feature_names=iris.feature_names)
'''
将决策树可视化函数生成的文件中的代码复制到www.webgraphviz.com显示决策树。
'''

'''
决策树算法总结：
1.优点：
简单的理解和解释，树木可视化。
2.缺点：
过于复杂的树不能很好地推广，这被称为过拟合。
3.改进：
①减枝cart算法(决策树API当中已经实现)
②随机森林
'''









