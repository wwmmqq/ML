package knnalgorithmclass.analyze.classify;

import org.wltea.analyzer.lucene.IKAnalyzer;

import java.io.*;
import java.text.DecimalFormat;
import java.util.*;

/**
 * Created by carson-pc on 2015/11/24.
 */
public class ComputeTFIDFClass
{
	/**
	 * @param args
	 */

	static IKAnalyzer analyzer = new IKAnalyzer();
	static ArrayList<String> stopWordsArray = new ArrayList<String>();

	public static void ReadStopWord(String path) throws IOException
	{
		FileReader stopWordsReader = new FileReader(path);
		String stopWordsLine;
		BufferedReader stopWordsBR = new BufferedReader(stopWordsReader);
		while ((stopWordsLine = stopWordsBR.readLine()) != null)
		{
			if (!stopWordsLine.isEmpty())
			{
				stopWordsArray.add(stopWordsLine);
			}
		}
	}

	private static ArrayList<String> FileList = new ArrayList<String>(); // the list of file

	//get list of file for the directory, including sub-directory of it
	public static List<String> readDirs(String filepath) throws FileNotFoundException, IOException
	{
		try
		{
			File file = new File(filepath);
			if (!file.isDirectory())
			{
				System.out.println("filepath:" + file.getAbsolutePath());
			}
			else
			{
				String[] flist = file.list();
				for (int i = 0; i < flist.length; i++)
				{
					File newfile = new File(filepath + "\\" + flist[i]);
					if (!newfile.isDirectory())
					{
						FileList.add(newfile.getAbsolutePath());
					}
					else if (newfile.isDirectory()) //if file is a directory, call ReadDirs
					{
						readDirs(filepath + "\\" + flist[i]);
					}
				}
			}
		}
		catch (FileNotFoundException e)
		{
			System.out.println(e.getMessage());
		}
		return FileList;
	}

	//read file
	public static String readFile(String file) throws FileNotFoundException, IOException
	{
		StringBuffer strSb = new StringBuffer(); //String is constant�� StringBuffer can be changed.
		InputStreamReader inStrR = new InputStreamReader(new FileInputStream(file), "gbk"); //byte streams to character streams
		BufferedReader br = new BufferedReader(inStrR);
		String line = br.readLine();
		while (line != null)
		{
			strSb.append(line).append("\r\n");
			line = br.readLine();
		}

		return strSb.toString();
	}

	//word segmentation
	public static ArrayList<String> cutWords(String file) throws IOException
	{

		ArrayList<String> words = new ArrayList<String>();
		String text = readFile(file);//��ȡ���ı��ַ���Ҫȥͣ�ô�
		words = analyzer.split(text);
		for (String word : stopWordsArray)
		{
			words.remove(word);
		}
		return words;
	}

	//term frequency in a file, times for each word
	public static HashMap<String, Integer> normalTF(ArrayList<String> cutwords)
	{
		HashMap<String, Integer> resTF = new HashMap<String, Integer>();

		for (String word : cutwords)
		{
			if (resTF.get(word) == null)
			{
				resTF.put(word, 1);
				System.out.println(word);
			}
			else
			{
				resTF.put(word, resTF.get(word) + 1);
				System.out.println(word.toString());
			}
		}
		return resTF;
	}

	//term frequency in a file, frequency of each word
	public static HashMap<String, Double> tf(ArrayList<String> cutwords)
	{
		HashMap<String, Double> resTF = new HashMap<String, Double>();

		int wordLen = cutwords.size();
		HashMap<String, Integer> intTF = normalTF(cutwords);

		Iterator iter = intTF.entrySet().iterator(); //iterator for that get from TF
		while (iter.hasNext())
		{
			Map.Entry entry = (Map.Entry) iter.next();
			resTF.put(entry.getKey().toString(), (Double.parseDouble(entry.getValue().toString()) / wordLen));
			//System.out.println(entry.getKey().toString() + " = " + Float.parseFloat(entry.getValue().toString()) / wordLen);
		}
		return resTF;
	}

	//tf times for file
	public static HashMap<String, HashMap<String, Integer>> normalTFAllFiles(String dirc) throws IOException
	{
		HashMap<String, HashMap<String, Integer>> allNormalTF = new HashMap<String, HashMap<String, Integer>>();

		List<String> filelist = readDirs(dirc);
		for (String file : filelist)
		{
			HashMap<String, Integer> dict = new HashMap<String, Integer>();
			ArrayList<String> cutwords = cutWords(file); //get cut word for one file

			dict = normalTF(cutwords);
			allNormalTF.put(file, dict);
		}
		return allNormalTF;
	}

	//tf for all file
	public static HashMap<String, HashMap<String, Double>> tfAllFiles(String dirc) throws IOException
	{
		HashMap<String, HashMap<String, Double>> allTF = new HashMap<String, HashMap<String, Double>>();
		List<String> filelist = readDirs(dirc);

		for (String file : filelist)
		{
			HashMap<String, Double> dict = new HashMap<String, Double>();
			ArrayList<String> cutwords = cutWords(file); //get cut words for one file

			dict = tf(cutwords);
			allTF.put(file, dict);
		}
		return allTF;
	}

	public static HashMap<String, Double> idf(HashMap<String, HashMap<String, Double>> all_tf)
	{
		HashMap<String, Double> resIdf = new HashMap<String, Double>();
		HashMap<String, Integer> dict = new HashMap<String, Integer>();
		int docNum = FileList.size();

		for (int i = 0; i < docNum; i++)
		{
			HashMap<String, Double> temp = all_tf.get(FileList.get(i));
			Iterator iter = temp.entrySet().iterator();
			while (iter.hasNext())
			{
				Map.Entry entry = (Map.Entry) iter.next();
				String word = entry.getKey().toString();
				if (dict.get(word) == null)
				{
					dict.put(word, 1);
				}
				else
				{
					dict.put(word, dict.get(word) + 1);
				}
			}
		}
		System.out.println("IDF for every word is:");
		Iterator iter_dict = dict.entrySet().iterator();
		while (iter_dict.hasNext())
		{
			Map.Entry entry = (Map.Entry) iter_dict.next();
			double tmp = (double) Math.log(docNum / Float.parseFloat(entry.getValue().toString()));
			double value = (double) Math.round(tmp * 10000) / 10000;
			resIdf.put(entry.getKey().toString(), value);
			//System.out.println(entry.getKey().toString() + " = " + value);
		}
		return resIdf;
	}

	public static void tf_idf(HashMap<String, HashMap<String, Double>> all_tf, HashMap<String, Double> idfs) throws IOException
	{
		HashMap<String, HashMap<String, Double>> resTfIdf = new HashMap<String, HashMap<String, Double>>();

		int docNum = FileList.size();
		for (int i = 0; i < docNum; i++)
		{
			String filepath = FileList.get(i);
			HashMap<String, Double> tfidf = new HashMap<String, Double>();
			HashMap<String, Double> temp = all_tf.get(filepath);
			Iterator iter = temp.entrySet().iterator();
			while (iter.hasNext())
			{
				Map.Entry entry = (Map.Entry) iter.next();
				String word = entry.getKey().toString();
				double tmp = Float.parseFloat(entry.getValue().toString()) * idfs.get(word);
				double value = (double) Math.round(tmp * 1000000) / 1000000;
				tfidf.put(word, value);
			}
			resTfIdf.put(filepath, tfidf);
		}
		System.out.println("TF-IDF for Every file is :");
		writeToTrainFile(resTfIdf);
		//DisTfIdf(resTfIdf);
	}

	/*
	public static void DisTfIdf(HashMap<String, HashMap<String, Float>> tfidf)
	{
		Iterator iter1 = tfidf.entrySet().iterator();
		while (iter1.hasNext())
		{
			Map.Entry entrys = (Map.Entry) iter1.next();
			System.out.println("FileName: " + entrys.getKey().toString());
			System.out.print("{");
			HashMap<String, Float> temp = (HashMap<String, Float>) entrys.getValue();
			Iterator iter2 = temp.entrySet().iterator();
			while (iter2.hasNext())
			{
				Map.Entry entry = (Map.Entry) iter2.next();
				System.out.print(entry.getKey().toString() + " = " + entry.getValue().toString() + ", ");
			}
			System.out.println("}");
		}

	}
	*/

	public static void writeToTrainFile(HashMap<String, HashMap<String, Double>> tfidf) throws IOException
	{
		String trainfilename = "generatedTrain.csv";
		FileWriter fw = new FileWriter(trainfilename, true);
		BufferedWriter bw = new BufferedWriter(fw);
		PrintWriter pw = new PrintWriter(bw, true);
		Iterator iter1 = tfidf.entrySet().iterator();
		while (iter1.hasNext())
		{
			Map.Entry entrys = (Map.Entry) iter1.next();
			//�����ǩ
			pw.print(entrys.getKey().toString().split("\\\\")[4].toString());
			HashMap<String, Float> temp = (HashMap<String, Float>) entrys.getValue();
			Iterator iter2 = temp.entrySet().iterator();
			int count = 0;
			while (iter2.hasNext())
			{
				Map.Entry entry = (Map.Entry) iter2.next();
				pw.print("," + String.format("%.7f", entry.getValue()));
				//System.out.print(entry.getKey().toString() + " = " + entry.getValue().toString() + ", ");
				count++;
				if (count > 30)
					break;
			}
			pw.println();
		}
		pw.flush();
		pw.close();
		System.out.println("write success");
	}

	public static void main(String[] args) throws IOException
	{
		// TODO Auto-generated method stub
		String file = "D:\\KNN_Class\\KNN_Data\\littledata_train";
		ReadStopWord("D:\\RiousSvn_Code\\LDAModel_GibbsSampling\\src\\knnalgorithmclass\\stopwords.txt");
		HashMap<String, HashMap<String, Double>> all_tf = tfAllFiles(file);
		System.out.println();
		HashMap<String, Double> idfs = idf(all_tf);
		System.out.println();
		tf_idf(all_tf, idfs);

	}
}
