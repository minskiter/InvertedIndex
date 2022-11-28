package com.example;

import java.io.BufferedReader;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.stream.Collectors;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.lib.partition.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.jobcontrol.ControlledJob;
import org.apache.hadoop.mapreduce.lib.jobcontrol.JobControl;
import org.apache.hadoop.util.GenericOptionsParser;

import javafx.util.Pair;

public class InvertIndex2 {

  public static class WordDocId implements WritableComparable<WordDocId> {

    private String word;
    private String docId;

    // get and set
    public String getWord() {
      return word;
    }

    public String getDocId() {
      return docId;
    }

    public void setWord(String word) {
      this.word = word;
    }

    public void setDocId(String docId) {
      this.docId = docId;
    }

    @Override
    public void write(DataOutput out) throws IOException {
      out.writeUTF(this.word);
      out.writeUTF(this.docId);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
      this.word = in.readUTF();
      this.docId = in.readUTF();
    }

    @Override
    public String toString() {
      return this.word + '\t' + this.docId;
    }

    @Override
    public int compareTo(WordDocId o) {
      if (this.word.equals(o.getWord())) {
        return this.docId.compareTo(o.getDocId());
      }
      return this.word.compareTo(o.getWord());
    }

  }

  public static class WordPartitioner extends HashPartitioner<WordDocId, LongWritable> {
    @Override
    public int getPartition(WordDocId key, LongWritable value, int numReduceTasks) {
      return Math.abs(key.getWord().hashCode()) % numReduceTasks;
    }
  }

  public static class TokenizerMapper
      extends Mapper<LongWritable, Text, WordDocId, LongWritable> {

    private WordDocId wordDocId = new WordDocId();
    private LongWritable one = new LongWritable(1);

    private int lexiconTreeLength = 1;
    private HashMap<Character, Integer>[] lexiconTree = new HashMap[500000];
    private HashSet<Integer> isWord = new HashSet<>();
    private Integer[] failto = new Integer[500000];
    private Queue<Integer> queue = new LinkedList<>();
    private Integer[] depth = new Integer[500000];

    public void setup(Context context) throws IOException {
      // 读取字典，创建字典树
      Configuration configuration = context.getConfiguration();
      FileSystem fs = FileSystem.get(configuration);
      FSDataInputStream input = fs.open(new Path("/dict/dict.txt"));
      BufferedReader reader = new BufferedReader(new InputStreamReader(input));
      String line = reader.readLine();
      lexiconTree[0] = new HashMap<Character, Integer>();
      while (line != null) {
        String word = line.split(" ")[0];
        int pointer = 0;
        for (int i = 0; i < word.length(); ++i) {
          Character ch = word.charAt(i);
          if (!lexiconTree[pointer].containsKey(ch)) {
            lexiconTree[pointer].put(ch, lexiconTreeLength);
            lexiconTree[lexiconTreeLength] = new HashMap<Character, Integer>();
            ++lexiconTreeLength;
          }
          pointer = lexiconTree[pointer].get(ch);
        }
        isWord.add(pointer);
        line = reader.readLine();
      }
      // 构建KMP失败回溯节点（AC自动机）
      int root = 0;
      depth[root] = 0;
      for (Character ch : lexiconTree[root].keySet()) {
        Integer cur = lexiconTree[root].get(ch);
        queue.add(cur);
        failto[cur] = root;
        depth[cur] = 1;
      }
      while (!queue.isEmpty()) {
        Integer p = queue.poll();
        for (Character ch : lexiconTree[p].keySet()) {
          Integer fail = failto[p];
          Integer cur = lexiconTree[p].get(ch);
          while (true) {
            if (lexiconTree[fail].containsKey(ch)) {
              failto[cur] = lexiconTree[fail].get(ch);
              break;
            } else {
              if (fail == 0) {
                failto[cur] = fail;
                break;
              }
              // 尝试再上一个失败节点
              fail = failto[fail];
            }
          }
          depth[cur] = depth[p] + 1;
          queue.add(cur);
        }
      }
    }

    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
      String[] ctx = value.toString().split("\t");
      if (ctx.length == 2) {
        wordDocId.setDocId(ctx[0]);
        String sentence = ctx[1];
        int pointer = 0;
        for (int index = 0; index < sentence.length(); ++index) {
          Character ch = sentence.charAt(index);
          while (pointer != 0 && !lexiconTree[pointer].containsKey(ch)) {
            pointer = failto[pointer];
          }
          if (lexiconTree[pointer].containsKey(ch)) {
            pointer = lexiconTree[pointer].get(ch);
            int p = pointer;
            while (p != 0) {
              if (isWord.contains(p)) {
                int r = index + 1;
                int l = r - depth[p];
                wordDocId.setWord(sentence.substring(l, r));
                context.write(wordDocId, one);
              }
              p = failto[p];
            }
          }
        }
      }
    }
  }

  public static class CounterReducer
      extends Reducer<WordDocId, LongWritable, WordDocId, LongWritable> {

    private LongWritable result = new LongWritable();

    public void reduce(WordDocId key, Iterable<LongWritable> values,
        Context context) throws IOException, InterruptedException {
      long sum = 0;
      for (LongWritable value : values) {
        sum += value.get();
      }
      result.set(sum);
      context.write(key, result);
    }
  }

  public static class DocIdFrequency implements Writable {

    private String docId;
    private Long frequency;

    public void setDocId(String docId) {
      this.docId = docId;
    }

    public String getDocId() {
      return docId;
    }

    public void setFrequency(Long frequency) {
      this.frequency = frequency;
    }

    public Long getFrequency() {
      return frequency;
    }

    @Override
    public void write(DataOutput out) throws IOException {
      out.writeUTF(this.docId);
      out.writeLong(this.frequency);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
      this.docId = in.readUTF();
      this.frequency = in.readLong();
    }

    @Override
    public String toString() {
      return "(" + this.docId + "," + Long.toString(this.frequency) + ')';
    }

    public DocIdFrequency clone() {
      DocIdFrequency docIdFrequency = new DocIdFrequency();
      docIdFrequency.setDocId(docId);
      docIdFrequency.setFrequency(frequency);
      return docIdFrequency;
    }

  }

  public static class DocIdFrequencyArray implements Writable {

    private ArrayList<DocIdFrequency> data;

    public ArrayList<DocIdFrequency> getData() {
      return data;
    }

    public void init() {
      if (data == null) {
        data = new ArrayList<>();
      }
    }

    public void setData(ArrayList<DocIdFrequency> data) {
      this.data = data;
    }

    public void add(DocIdFrequency docIdFrequency) {
      init();
      data.add(docIdFrequency);
    }

    public boolean isEmpty() {
      return size() == 0;
    }

    public int size() {
      if (data == null)
        return 0;
      return data.size();
    }

    public void clear() {
      if (data == null)
        return;
      data.clear();
    }

    public DocIdFrequency get(Integer index) throws IndexOutOfBoundsException {
      if (data == null) {
        throw new IndexOutOfBoundsException("data is null");
      }
      return data.get(index);
    }

    public DocIdFrequencyArray clone() {
      if (data == null) {
        return null;
      }
      DocIdFrequencyArray array = new DocIdFrequencyArray();
      for (DocIdFrequency docIdFrequency : data) {
        array.add(docIdFrequency.clone());
      }
      return array;
    }

    @Override
    public void write(DataOutput out) throws IOException {
      int length = 0;
      if (data != null) {
        length = data.size();
      }
      out.writeInt(length);
      if (data != null)
        for (DocIdFrequency docIdFrequency : data) {
          docIdFrequency.write(out);
        }
    }

    @Override
    public void readFields(DataInput in) throws IOException {
      int length = in.readInt();
      data = new ArrayList<DocIdFrequency>();
      for (int i = 0; i < length; ++i) {
        DocIdFrequency docIdFrequency = new DocIdFrequency();
        docIdFrequency.readFields(in);
        data.add(docIdFrequency);
      }
    }

    public String toString() {
      if (data == null)
        return "";
      return data.stream().map(DocIdFrequency::toString).collect(Collectors.joining(", "));
    }

  }

  public static class WordIndexMapper extends Mapper<LongWritable, Text, Text, DocIdFrequencyArray> {
    private Text word = new Text();
    private DocIdFrequency value = new DocIdFrequency();
    private DocIdFrequencyArray array = new DocIdFrequencyArray();

    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
      String[] values = value.toString().split("\t");
      if (array.isEmpty())
        array.add(this.value);
      if (values.length == 3) {
        this.word.set(values[0]);
        this.value.setDocId(values[1]);
        this.value.setFrequency(Long.parseLong(values[2]));
        context.write(this.word, array);
      }
    }
  }

  public static class InvertIndexReducer extends Reducer<Text, DocIdFrequencyArray, Text, DocIdFrequencyArray> {

    public void reduce(Text key, Iterable<DocIdFrequencyArray> values, Context context)
        throws IOException, InterruptedException {
      ArrayList<DocIdFrequencyArray> array = new ArrayList<DocIdFrequencyArray>();
      for (DocIdFrequencyArray value : values) {
        array.add(value.clone());
      }

      Queue<Pair<Integer, Integer>> queue = new PriorityQueue<Pair<Integer, Integer>>((a, b) -> {
        if (a.getKey() == b.getKey()) {
          return 0;
        }
        return a.getKey() - b.getKey();
      });
      for (int i = 0; i < array.size(); ++i) {
        queue.add(new Pair<Integer, Integer>(array.get(i).size(), i));
      }
      // 按长度最小的序列进行归并排序
      while (queue.size() > 1) {
        DocIdFrequencyArray temp = new DocIdFrequencyArray();
        Pair<Integer, Integer> first = queue.poll();
        Pair<Integer, Integer> second = queue.poll();
        int l = 0, r = 0;
        while (l < first.getKey() && r < second.getKey()) {
          if (array.get(first.getValue()).get(l).getFrequency() > array.get(second.getValue()).get(r).getFrequency()) {
            temp.add(array.get(first.getValue()).get(l));
            ++l;
          } else {
            temp.add(array.get(second.getValue()).get(r));
            ++r;
          }
        }
        while (l < first.getKey()) {
          temp.add(array.get(first.getValue()).get(l));
          ++l;
        }
        while (r < second.getKey()) {
          temp.add(array.get(second.getValue()).get(r));
          ++r;
        }
        array.set(first.getValue(), temp);
        queue.add(new Pair<Integer, Integer>(temp.size(), first.getValue()));
      }

      context.write(key, array.get(queue.poll().getValue()));
    }
  }

  public static void main(String[] args) throws Exception {

    Configuration conf = new Configuration();
    String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
    if (otherArgs.length < 2) {
      System.err.println("Usage: com.example.InvertIndex <in> [<in>...] <out>");
      System.exit(2);
    }

    FileSystem fileSystem = FileSystem.get(conf);

    // clear output directory
    fileSystem.delete(new Path(otherArgs[otherArgs.length - 1]), true);
    fileSystem.delete(new Path("/job1"), true);

    // job1
    Job job1 = Job.getInstance(conf, "Word Frequency Counter");

    job1.setJarByClass(InvertIndex.class);
    job1.setMapperClass(TokenizerMapper.class);

    job1.setMapOutputKeyClass(WordDocId.class);
    job1.setMapOutputValueClass(LongWritable.class);

    job1.setCombinerClass(CounterReducer.class);
    job1.setReducerClass(CounterReducer.class);
    job1.setPartitionerClass(WordPartitioner.class);
    job1.setNumReduceTasks(2);

    job1.setOutputKeyClass(WordDocId.class);
    job1.setOutputValueClass(LongWritable.class);
    for (int i = 0; i < otherArgs.length - 1; ++i) {
      FileInputFormat.addInputPath(job1, new Path(otherArgs[i]));
    }
    FileOutputFormat.setOutputPath(job1,
        new Path("/job1"));

    job1.waitForCompletion(true);

    // job2
    Job job2 = Job.getInstance(conf, "Inverted Index");

    job2.setJarByClass(InvertIndex.class);
    job2.setMapperClass(WordIndexMapper.class);
    job2.setMapOutputKeyClass(Text.class);
    job2.setMapOutputValueClass(DocIdFrequencyArray.class);

    job2.setCombinerClass(InvertIndexReducer.class);
    job2.setReducerClass(InvertIndexReducer.class);
    job2.setNumReduceTasks(2);

    job2.setOutputKeyClass(Text.class);
    job2.setOutputValueClass(DocIdFrequencyArray.class);
    FileInputFormat.addInputPath(job2, new Path("/job1"));

    FileOutputFormat.setOutputPath(job2, new Path(otherArgs[otherArgs.length - 1]));

    job2.waitForCompletion(true);

  }
}