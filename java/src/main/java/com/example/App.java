package com.example;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Queue;


public class App 
{
    private HashMap<Character, Integer>[] lexiconTree = new HashMap[500000];
    private HashSet<Integer> isWord = new HashSet<>();
    private Integer[] failto = new Integer[500000];
    private Queue<Integer> queue = new LinkedList<>();
    private Integer[] depth = new Integer[500000];
    private Integer lexiconTreeLength = 1;

    void run() throws FileNotFoundException,IOException{
        BufferedReader reader = new BufferedReader(new FileReader("./temp/test.txt"));
        String line = reader.readLine();
        lexiconTree[0] = new HashMap<>();
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
        for (Character ch:lexiconTree[root].keySet()){
            Integer cur = lexiconTree[root].get(ch);
            queue.add(cur);
            failto[cur] = root;
            depth[cur] = 1;
        }
        while (!queue.isEmpty()){
            Integer p = queue.poll();
            for (Character ch:lexiconTree[p].keySet()){
                Integer fail = failto[p];
                Integer cur = lexiconTree[p].get(ch);
                while (true){
                    if (lexiconTree[fail].containsKey(ch)){
                        failto[cur] = lexiconTree[fail].get(ch);
                        break;
                    }else{
                        if (fail==0){
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
        String s = "您好啊非常谢谢你";
        int pointer = 0;
        ArrayList<String> ans = new ArrayList<>();
        for (int index=0;index<s.length();++index){
            Character ch = s.charAt(index);
            while (pointer!=0 && !lexiconTree[pointer].containsKey(ch)){
                pointer = failto[pointer];
            }
            System.out.println(pointer);
            if (lexiconTree[pointer].containsKey(ch)){
                pointer = lexiconTree[pointer].get(ch);
                int p = pointer;
                while (p!=0){
                    if (isWord.contains(p)){
                        int r = index+1;
                        int l = r-depth[p];
                        System.out.println(l+","+r);
                        ans.add(s.substring(l, r));
                    }
                    p = failto[p];
                }
            }
        }
        for (String string : ans) {
            System.out.println(string);
        }
    }

    public static void main( String[] args ) throws FileNotFoundException,IOException
    {
        new App().run();
    }
}
