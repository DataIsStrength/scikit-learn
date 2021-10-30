# -*- coding: utf-8 -*-
"""
文本特征抽取
"""

#导入向量化类
from sklearn.feature_extraction.text import CountVectorizer

#输入文本
data=['life is short,i like python',
      'life is too long,i dislike python']
#实例化向量化类并设置停用词
transfer=CountVectorizer(stop_words=['is','too'])
#调用fit_transform
data=transfer.fit_transform(data)
#.toarray()函数使data由sparse矩阵转换为二位数组形式
print('文本特征抽取的结果:\n', data.toarray())
print('返回特征名字:\n', transfer.get_feature_names())
'''
在抽取本文特征时，将单词作为特征，
标点符号和单个的字母被认为没有具体的意义，所以被自动忽略。
'''