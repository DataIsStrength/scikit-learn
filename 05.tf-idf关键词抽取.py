# -*- coding: utf-8 -*-
"""
tf-idf关键词抽取
"""

#导入tf-idf向量化类
from sklearn.feature_extraction.text import TfidfVectorizer
#导入jieba包
import jieba

#定义分词函数
def cut_words(text):
    #用jieba对中文字符串进行分词并用join函数转化为字符串形式
    text=' '.join(list(jieba.cut(text)))
    return text

#输入文本
data=['一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。',
      '我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。',
      '如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。']

#逐句进行分词
text_list=[]
for sent in data:
    text_list.append(cut_words(sent))
print(text_list)

#实例化向量化类
transfer=TfidfVectorizer(stop_words=['一种','不会','不要'])
#调用fit_transform
data=transfer.fit_transform(text_list)
#.toarray()函数使data由sparse矩阵转换为二维数组形式
print('文本特征抽取的结果:\n', data.toarray())
print('返回特征名字:\n', transfer.get_feature_names())
'''
tf-idf向量化函数返回的是词语的tf-idf值
tf-idf=tf*idf，其中
tf即词频（term frequency，tf），
指的是某一个给定的词语在该文件中出现的频率；
idf即逆向文档频率（inverse document frequency，idf），
是一个词语普遍重要性的度量，某一特定词语的idf，
可以由总文件数目除以包含该词语之文件的数目，
再将得到的商取以10为底的对数得到。
tf-idf值可以反映一个词语的关键程度。
'''