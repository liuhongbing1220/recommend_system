
import org.apache.spark.sql.{ SparkSession, _ }
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.feature.Word2Vec
import java.util.Date
import java.text.SimpleDateFormat
import org.apache.spark.{SparkConf, SparkContext}

object Word2vec {

  /**
   * word2vec实现：
   *
   * 1）读取训练样本
   * 2）w2v模型训练
   * 3）提取词向量，并且计算相似词
   *
   * @author sunbow
   */


  def word2VecRun(sc:SparkContext) = {

    val path = "/Users/liuhongbing/Documents/data/zuowen/"
    val input = sc.textFile(path+"data_seg_result").map(line => line.split(" ").toSeq)
    //model train
    val word2vec = new Word2Vec().setVectorSize(50)

    val model = word2vec.fit(input)
    println("model word size: " + model.getVectors.size)

    //Save and load model
    model.save(sc, path + "data_w2v")

    val local = model.getVectors.map{
      case (word, vector) => Seq(word, vector.mkString(" ")).mkString(":")
    }.toArray
    sc.parallelize(local).saveAsTextFile(path+"data_w2v_embedding")

    //predict similar words
    val like = model.findSynonyms("中国", 40)
    for ((item, cos) <- like) {
      println(s"$item  $cos")
    }

  }


  def main(args: Array[String]): Unit ={
    val conf = new SparkConf().setAppName("word2vec").setMaster("local")
    val sc = new SparkContext(conf)
    word2VecRun(sc)
    sc.stop()
  }

}