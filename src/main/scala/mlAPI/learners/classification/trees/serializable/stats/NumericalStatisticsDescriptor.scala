package mlAPI.learners.classification.trees.serializable.stats

import com.fasterxml.jackson.core.JsonProcessingException
import com.fasterxml.jackson.databind.ObjectMapper

/**
 * A serializable descriptor of a numerical statistics instance.
 */
class NumericalStatisticsDescriptor(classCountersKeys: Array[Int],
                                    classCountersValues: Array[Double],
                                    max: Double,
                                    prediction: Int,
                                    n_l: Double,
                                    isActive: Boolean,
                                    var attributeNormalsDescriptor: Array[AttributeGaussianDescriptor],
                                    var dropped: Array[Int],
                                    var splits: Int)
  extends StatisticsDescriptor(classCountersKeys, classCountersValues, max, prediction, n_l, isActive)
    with java.io.Serializable {

  def setAttributeNormalsDescriptor(attributeNormalsDescriptor: Array[AttributeGaussianDescriptor]): Unit =
    this.attributeNormalsDescriptor = attributeNormalsDescriptor

  def getAttributeNormalsDescriptor: Array[AttributeGaussianDescriptor] = attributeNormalsDescriptor

  def setDropped(dropped: Array[Int]): Unit = this.dropped = dropped

  def getDropped: Array[Int] = dropped

  def setSplits(splits: Int): Unit = this.splits = splits

  def getSplits: Int = splits

  override def toString: String = {
    try {
      toJsonString
    } catch {
      case _: JsonProcessingException => "Non printable " + this.getClass.getName
    }
  }

  override def toJsonString: String = {
    new ObjectMapper().writerWithDefaultPrettyPrinter().writeValueAsString(this)
  }

}