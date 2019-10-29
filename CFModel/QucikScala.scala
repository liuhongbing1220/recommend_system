import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object QuickScala {

  /**
   *
   * @param X
   * @return
   */
  def sign(X:Int):Int={
    var flag = 0
    if(X > 0) {
      flag = X
    }
    if(X < 0) {
      flag = -1 * X
    }

    flag
  }

  /**
   * 倒序输出
   * @param X
   */
  def reverse_print(X:Int): Unit={
    for(a <- X to 1 by -1){
      println(a)
    }
  }

  /**
   * 计算字符串unicode乘积
   * @param Str
   * @return
   */
  def calcStringMulti(Str:String): Long ={
    var t = 1L
    for(a <- Str){
      t = t * a.toLong
    }
    t
  }

  /**
   * 实现 x^n, pow(x,n)
   * @param n
   * @param X
   * @return
   */
  def calcPowX(n:Int, X:Int):Long={

    if(n == 0)
      1
    else if(n == 1)
      X
    else if(n % 2 == 0)
      calcPowX(n/2, X)*calcPowX(n/2, X)
    else
      calcPowX(n-1, X) * X
  }


  /**
   * 生成随机数组
   * @param n
   * @return
   */
  def randArray(n:Int): ArrayBuffer[Int] ={
    val randarray = new ArrayBuffer[Int]()
    for(i <- 0 to n){
      randarray += Random.nextInt(n)
    }
    randarray
  }

  /**
   * 交换相邻位置
   * @param array
   * @return
   */
  def changeArray(array: Array[Int]): Array[Int]={
    val arraynew = new Array[Int](array.length)
    for(i <- 0 to array.length){
      if(i % 2 == 0 ){
        if(i < array.length) {
           arraynew(i) = array(i+1)
        }
      }else{
        arraynew(i) = array(i-1)
      }
    }
    arraynew
  }

  /**
   * 统计单词个数
   * @return
   */
  def wordCount()={
    val str = "symbol is a mark, sign or word that indicates, signifies, or is understood as representing an idea, object, or relationship"
    val map_init = new scala.collection.mutable.HashMap[String, Int]
    for(i <- str.split(" ")){
      if(map_init.contains(i)){
        map_init(i) = map_init(i) + 1
      }else{
        map_init(i) = 1
      }
    }
    map_init
  }

  /**
   * 辅助构造器
   */
  class Person1{
    private var name = ""
    private var age = 0
    def this(name:String){
      this()
      this.name = name
    }

    def this(name:String, age:Int){
      this(name)
      this.age = age
    }

  }

  /**
   * 主构造器
   * @param args
   */
  class Person2(name:String,age:Int){
    def describe()={
      name + " is " + age +" years old"
    }
  }


  /**
   * 账号存取钱
   */
  class Bank{
    private var balanced = 0
    def deposit(money:Int): Unit ={
      balanced += money
    }
    def withdraw(money:Int): Unit ={
      balanced -= money
    }
    def current = balanced
  }

  def main(args:Array[String]): Unit ={

    val bank = new Bank()
    println(bank.current)
    bank.deposit(100)
    println(bank.current)
    bank.withdraw(25)
    println(bank.current)

//    val person2 = new Person2("liuhongbing",29)
//    println(person2.describe())
//    val cou = new counter()
//    cou.increment()
//    println(cou.current)
//    println(cou.value)
//    println("-----------------")
//    cou.value = 10
//    wordCount().map(println)
//    print(sign(-20))
//    print("----------")
//    reverse_print(10)
//    print("--------")
//    print(calcStringMulti("Hello"))
//    println("--------")
//    println(calcPowX(4, 3))
//    println("--------")
//    randArray(10).map(println)
//    println("--------")
//    val array = Array(0,1,2,3,4,5,6,7)
//    changeArray(array).map(println)
//    println(array.sum/array.length)
//
//    val map_val = scala.collection.mutable.Map("alice"->10, "Bolb"->3, "cindy"->6)
//    println(map_val("alice"))
//    map_val("alice") = 11
//
//    map_val += ("liu"->20, "hong"->30)
//    map_val.map(println)
//
//    for ((k, v) <- map_val){
//      println(k,"\t",v)
//    }
//
//    map_val.keySet.map(println)
//    map_val.values.map(println)
  }
}