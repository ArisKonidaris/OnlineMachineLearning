package mlAPI.learners.classification.trees.serializable.stats

import com.fasterxml.jackson.core.JsonProcessingException
import com.fasterxml.jackson.databind.ObjectMapper

/**
 * A serializable descriptor of a gaussian distribution.
 */
class GaussianDescriptor(var mean: Double, var d_squared: Double, var count: Long) extends java.io.Serializable {

  def setMean(mean: Double): Unit = this.mean = mean

  def getMean: Double = mean

  def setDSquared(d_squared: Double): Unit = this.d_squared = d_squared

  def getDSquared: Double = d_squared

  def setCount(count: Long): Unit = this.count = count

  def getCount: Long = count

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

}