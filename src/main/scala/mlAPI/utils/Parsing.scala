package mlAPI.utils

import scala.collection.mutable

object Parsing {

  def DoubleParsing(config: mutable.Map[String, AnyRef], name: String, default: Double): Double = {
    try {
      if (config(name).isInstanceOf[Double])
        config(name).asInstanceOf[Double]
      else if (config(name).isInstanceOf[Int])
        config(name).asInstanceOf[Int].toDouble
      else
        default
    } catch {
      case _: Throwable => default
    }
  }

  def IntegerParsing(config: mutable.Map[String, AnyRef], name: String, default: Int): Int = {
    try {
      if (config(name).isInstanceOf[Double])
        config(name).asInstanceOf[Double].toInt
      else if (config(name).isInstanceOf[Int])
        config(name).asInstanceOf[Int]
      else
        default
    } catch {
      case _: Throwable => default
    }
  }

}
