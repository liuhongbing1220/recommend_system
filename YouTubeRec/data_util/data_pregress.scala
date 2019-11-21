
val user_rating_path = "/Users/liuhongbing/Documents/data/recommend/ml-20m/ratings.csv"
val user_rating_df = spark.read.options(Map(("delimiter", ","), ("header", "true"))).csv(user_rating_path)

val ml_item = user_rating_df.select("movieId").distinct
ml_item.coalesce(1).write.mode(SaveMode.Overwrite).format("com.databricks.spark.csv").save("/Users/liuhongbing/Documents/data/recommend/ml-20m/ml_Items")

import org.apache.spark.sql.types._
val user_rating_df2 = user_rating_df.withColumn("timestamp_Int", col("timestamp").cast(IntegerType))
+-------+--------------------+                                                  
|summary|       timestamp_Int|
+-------+--------------------+
|  count|            20000263|
|   mean|1.1009179216771157E9|
| stddev|1.6216942478273147E8|
|    min|           789652004|
|    max|          1427784002|
+-------+--------------------+

case class ItemPref(
                     val userid : String,
                     val itemid : String,
                     val pref : Double,
                     val timestamp: String) extends Serializable


val user_ds = user_rating_df.map {
    case Row(userId: String, movieId: String, rating: String, timestamp: String) =>
    ItemPref(userId, movieId, rating.toDouble,timestamp)
}

    // 1 数据做准备
val user_ds1 = user_ds.
    withColumn("iv", concat_ws(":", $"itemid", $"timestamp")).
    groupBy("userid").agg(collect_set("iv")).
    withColumnRenamed("collect_set(iv)", "itemid_set").
    select("userid", "itemid_set")

import scala.collection.mutable.WrappedArray


val user_ds2 = user_ds1.map{
    case Row(userid:String, itemid_set:WrappedArray[String]) => 
      val c = userid
      val ss = itemid_set
      val sortSorce = ss.toList.sortWith(_.split(':')(1).toDouble< _.split(":")(1).toDouble).mkString("||")
      (c, sortSorce)
}.withColumnRenamed("_1", "userId").withColumnRenamed("_2", "itemid_timstamp")


val randomSplitDs = user_ds2.randomSplit(Array(0.7,0.3), 100)

randomSplitDs(0).coalesce(1).write.mode(SaveMode.Overwrite).format("com.databricks.spark.csv").save("/Users/liuhongbing/Documents/data/recommend/ml-20m/ml_userInfo")
randomSplitDs(1).coalesce(1).write.mode(SaveMode.Overwrite).format("com.databricks.spark.csv").save("/Users/liuhongbing/Documents/data/recommend/ml-20m/ml_userInfo")
