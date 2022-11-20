package com.example;

import java.io.BufferedReader;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.stream.Collectors;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.jobcontrol.ControlledJob;
import org.apache.hadoop.mapreduce.lib.jobcontrol.JobControl;
import org.apache.hadoop.util.GenericOptionsParser;

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

  public static class TokenizerMapper
      extends Mapper<LongWritable, Text, WordDocId, LongWritable> {

    private WordDocId wordDocId = new WordDocId();
    private LongWritable one = new LongWritable(1);

    private int lexiconTreeLength = 1;
    private HashMap<Character, Integer>[] lexiconTree = new HashMap[500000];
    private HashSet<Integer> isWord = new HashSet<Integer>();

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
          if (!lexiconTree[pointer].containsKey(word.charAt(i))) {
            lexiconTree[pointer].put(word.charAt(i), lexiconTreeLength);
            lexiconTree[lexiconTreeLength] = new HashMap<Character,Integer>();
            ++lexiconTreeLength;
          }
          pointer = lexiconTree[pointer].get(word.charAt(i));
        }
        isWord.add(pointer);
        line = reader.readLine();
      }
    }

    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
      String[] ctx = value.toString().split("\t");
      if (ctx.length == 2) {
        wordDocId.setDocId(ctx[0]);
        String sentence = ctx[1];
        for (int index = 0; index < sentence.length(); ++index) {
          int pointer = 0;
          int r = index;
          ArrayList<String> list = new ArrayList<String>();
          StringBuffer buffer = new StringBuffer();
          while (r < sentence.length() && lexiconTree[pointer].containsKey(sentence.charAt(r))) {
            buffer.append(sentence.charAt(r));
            pointer = lexiconTree[pointer].get(sentence.charAt(r));
            if (isWord.contains(pointer)) {
              list.add(buffer.toString());
            }
            ++r;
          }
          // if (list.isEmpty()) {
          //   list.add(Character.toString(sentence.charAt(index)));
          // }
          for (String word : list) {
            wordDocId.setWord(word);
            context.write(wordDocId, one);
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

    public DocIdFrequency clone(){
      DocIdFrequency docIdFrequency = new DocIdFrequency();
      docIdFrequency.setDocId(docId);
      docIdFrequency.setFrequency(frequency);
      return docIdFrequency;
    }

  }

  public static class WordIndexMapper extends Mapper<LongWritable, Text, Text, DocIdFrequency> {
    private Text word = new Text();
    private DocIdFrequency value = new DocIdFrequency();

    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
      String[] values = value.toString().split("\t");
      if (values.length == 3) {
        this.word.set(values[0]);
        this.value.setDocId(values[1]);
        this.value.setFrequency(Long.parseLong(values[2]));
        context.write(this.word, this.value);
      }
    }
  }

  public static class InvertIndexReducer extends Reducer<Text, DocIdFrequency, Text, Text> {
    private Text text = new Text();

    public void reduce(Text key, Iterable<DocIdFrequency> values, Context context) throws IOException, InterruptedException {
      ArrayList<DocIdFrequency> array = new ArrayList<DocIdFrequency>();
      for (DocIdFrequency value : values) {
        array.add(value.clone());
      }
      array.sort((a,b)->{
        if (a.frequency==b.frequency) return 0;
        return a.frequency<b.frequency?1:-1;
      });
      text.set(array.stream().map(DocIdFrequency::toString).collect(Collectors.joining(", ")));
      context.write(key, text);
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
    Job job2 = Job.getInstance(conf, "Invert Index");

    job2.setJarByClass(InvertIndex.class);
    job2.setMapperClass(WordIndexMapper.class);
    job2.setMapOutputKeyClass(Text.class);
    job2.setMapOutputValueClass(DocIdFrequency.class);

    // job2.setCombinerClass(InvertIndexReducer.class);
    job2.setReducerClass(InvertIndexReducer.class);

    job2.setNumReduceTasks(2);

    FileInputFormat.addInputPath(job2, new Path("/job1"));

    FileOutputFormat.setOutputPath(job2, new Path(otherArgs[otherArgs.length - 1]));

    job2.waitForCompletion(true);
   
  }
}