package mlAPI.learners.classification.trees.serializable.stats

import ControlAPI.CountableSerial
import com.fasterxml.jackson.core.JsonProcessingException
import com.fasterxml.jackson.databind.ObjectMapper

/**
 * A serializable descriptor for a [[mlAPI.learners.classification..trees.stats.Statistics]] instance.
 */
class StatisticsDescriptor(var classCountersKeys: Array[Int],
                           var classCountersValues: Array[Double],
                           var max: Double,
                           var prediction: Int,
                           var n_l: Double,
                           var isActive: Boolean) extends CountableSerial {

  def setClassCountersKeys(classCountersKeys: Array[Int]): Unit = this.classCountersKeys = classCountersKeys

  def getClassCountersKeys: Array[Int] = classCountersKeys

  def setClassCountersValues(classCountersValues: Array[Double]): Unit = this.classCountersValues = classCountersValues

  def getClassCountersValues: Array[Double] = classCountersValues

  def setMax(max: Double): Unit = this.max = max

  def getMax: Double = max

  def setPrediction(prediction: Int): Unit = this.prediction = prediction

  def getPrediction: Int = prediction

  def setNl(n_l: Double): Unit = this.n_l = n_l

  def getNl: Double = n_l

  def setIsActive(isActive: Boolean): Unit = this.isActive = isActive

  def getIsActive: Boolean = isActive

  override def toString: String = {
    try {
      toJsonString
    } catch {
      case _: JsonProcessingException => "Non printable " + this.getClass.getName
    }
  }

  def toJsonString: String = {
    new ObjectMapper().writerWithDefaultPrettyPrinter().writeValueAsString(this)
  }

  override def getSize: Int = {
    { if (classCountersKeys != null) 4 * classCountersKeys.length else 0 } +
      { if (classCountersValues != null) 8 * classCountersValues.length else 0 } +
      21
  }

}
