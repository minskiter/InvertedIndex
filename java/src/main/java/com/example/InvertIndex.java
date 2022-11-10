package com.example;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
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

import com.huaban.analysis.jieba.*;
import com.huaban.analysis.jieba.JiebaSegmenter.SegMode;

public class InvertIndex {

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

    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
      JiebaSegmenter segmenter = new JiebaSegmenter();
      String[] ctx = value.toString().split("\t");
      if (ctx.length == 2) {
        wordDocId.setDocId(ctx[0]);
        List<SegToken> list = segmenter.process(ctx[1], SegMode.INDEX);
        for (SegToken token : list) {
          wordDocId.setWord(token.word);
          context.write(wordDocId, one);
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

  }

  public static class WordIndexMapper extends Mapper<LongWritable, Text, Text, Text> {
    private Text word = new Text();
    private Text text = new Text();
    private DocIdFrequency value = new DocIdFrequency();

    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
      String[] values = value.toString().split("\t");
      if (values.length == 3) {
        this.word.set(values[0]);
        this.value.setDocId(values[1]);
        this.value.setFrequency(Long.parseLong(values[2]));
        this.text.set(this.value.toString());
        context.write(this.word, this.text);
      }
    }
  }

  public static class InvertIndexReducer extends Reducer<Text, Text, Text, Text> {
    private Text text = new Text();

    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
      ArrayList<String> array = new ArrayList<String>();
      for (Text value : values) {
        String[] text = value.toString().split(", ");
        for (String t : text) {
          array.add(t);
        }
      }

      text.set(String.join(", ", array.toArray(new String[0])));
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

    

    JobControl ctl = new JobControl("InvertIndexGroup");

    // job1
    Job job1 = Job.getInstance(conf, "Word Frequency Counter");

    job1.setJarByClass(InvertIndex.class);
    job1.setMapperClass(TokenizerMapper.class);

    job1.setMapOutputKeyClass(WordDocId.class);
    job1.setMapOutputValueClass(LongWritable.class);

    job1.setCombinerClass(CounterReducer.class);
    job1.setReducerClass(CounterReducer.class);
    job1.setNumReduceTasks(16);

    job1.setOutputKeyClass(WordDocId.class);
    job1.setOutputValueClass(LongWritable.class);
    for (int i = 0; i < otherArgs.length - 1; ++i) {
      FileInputFormat.addInputPath(job1, new Path(otherArgs[i]));
    }
    FileOutputFormat.setOutputPath(job1,
        new Path("/job1"));

    ControlledJob cjob1 = new ControlledJob(conf);
    cjob1.setJob(job1);

    ctl.addJob(cjob1);

    // job2
    Job job2 = Job.getInstance(conf, "Invert Index");

    job2.setJarByClass(InvertIndex.class);
    job2.setMapperClass(WordIndexMapper.class);
    job2.setMapOutputKeyClass(Text.class);
    job2.setMapOutputValueClass(Text.class);

    job2.setCombinerClass(InvertIndexReducer.class);
    job2.setReducerClass(InvertIndexReducer.class);

    job2.setNumReduceTasks(16);

    FileInputFormat.addInputPath(job2, new Path("/job1"));

    FileOutputFormat.setOutputPath(job2, new Path(otherArgs[otherArgs.length - 1]));

    ControlledJob cjob2 = new ControlledJob(conf);
    cjob2.setJob(job2);
    cjob2.addDependingJob(cjob1);
    ctl.addJob(cjob2);

    Thread thread = new Thread(ctl);
    thread.start();

    while (true) {
      if (ctl.allFinished()) {
        ctl.stop();
        System.out.println(ctl.getSuccessfulJobList());
        System.out.println(ctl.getFailedJobList());
        fileSystem.delete(new Path("/job1"), true);
        System.exit(0);
      }
      if (ctl.getFailedJobList().size() > 0) {
        ctl.stop();
        System.out.println(ctl.getSuccessfulJobList());
        System.out.println(ctl.getFailedJobList());
        fileSystem.delete(new Path("/job1"), true);
        System.exit(1);
      }
      Thread.sleep(10);
    }
  }
}