### HADOOP 3 倒排索引小实验

#### 环境

- JAVA 1.8
- HADOOP 3.3.4
- SPARK 3.3.1

### JAVA(HADOOP)
``` sh
cd java
```

#### 包管理
- Maven

#### 编辑器

VSCode
- Extension Pack for JAVA
- Maven for JAVA

#### Dev

VSCode `ctrl+shift+B` 启动 tasks

1. 构建Fat JAR(build)

2. 发送到Hadoop上执行(run)

#### 输入目录
./input

输入格式:
documentId sentences

``` 
1 你好世界！
```

### PYTHON(SPARK)

#### 第三方包
./python/jieba.zip

#### 运行
VSCode `ctrl+shift+B` 启动 tasks

1. 运行(run)

#### 输入目录
hdfs://{hdfsip}:9000/input

#### 输出目录
hdfs://{hdfsip}:9000/output


