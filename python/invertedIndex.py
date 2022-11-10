#%%
from pyspark import SparkContext
from pyspark import SparkConf

conf = SparkConf().setAppName("InvertedIndex").setMaster("spark://master:7077")
sc = SparkContext(conf=conf).getOrCreate()
#%%
sc.addPyFile("jieba.zip")

#%%
rdds = sc.textFile("hdfs://master:9000/input/test.txt")

import jieba
def cut(s):
    docId,sentences = s.split("\t")
    words = list(jieba.cut_for_search(sentences))
    return list((word,docId) for word in words)

wordDocIdMapper = rdds.flatMap(cut)

#%%
wordDocIds = wordDocIdMapper.groupByKey()

#%%
from collections import Counter
def convert(x):
    return (x[0],list((k,v) for k,v in Counter(list(x[1])).items()))
invertedIndex = wordDocIds.map(convert)

#%%
print(invertedIndex.take(2))

#%%
invertedIndex.saveAsTextFile("hdfs://master:9000/output")

# %%
sc.stop()

