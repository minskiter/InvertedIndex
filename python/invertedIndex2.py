from pyspark import SparkContext
from pyspark import SparkConf

conf = SparkConf().setAppName("InvertedIndex").setMaster("spark://master:7077")
sc = SparkContext(conf=conf).getOrCreate()
sc.setLogLevel("INFO")
dicts = sc.textFile("hdfs://master:9000/dict/dict.txt")
words = dicts.map(lambda x:x.split()[0]).collect()
lexiconTree = [{}]
for word in words:
    p = 0
    for ch in word:
        if not ch in lexiconTree[p]:
            lexiconTree[p][ch] = len(lexiconTree)
            lexiconTree.append({})
        p = lexiconTree[p][ch]
    lexiconTree[p]["isWord"] = True
rdds = sc.textFile("hdfs://master:9000/input/test.txt")
def cut(s):
    global lexiconTree
    docId,sentences = s.split("\t")
    words = []
    for i in range(len(sentences)):
        p = 0
        r = i
        subwords = []
        while r<len(sentences) and sentences[r] in lexiconTree[p]:
            p = lexiconTree[p][sentences[r]]
            subwords.append(sentences[r])
            if "isWord" in lexiconTree[p]:
                words.append(''.join(subwords))
            r+=1
    return list((word,docId) for word in words)
wordDocIdMapper = rdds.flatMap(cut)
wordDocIds = wordDocIdMapper.groupByKey()
from collections import Counter
def convert(x):
    return (x[0],sorted(list((k,v) for k,v in Counter(list(x[1])).items()),key=lambda x:x[1],reverse=True))
invertedIndex = wordDocIds.map(convert)
invertedIndex.saveAsTextFile("hdfs://master:9000/output")
sc.stop()

