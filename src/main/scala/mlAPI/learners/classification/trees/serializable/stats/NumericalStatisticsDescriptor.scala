package mlAPI.learners.classification.trees.serializable.stats

import ControlAPI.CountableSerial
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
    with CountableSerial {

  override def getSize: Int = {
    25 +
      { if (classCountersKeys != null) 4 * classCountersKeys.length else 0 } +
      { if (classCountersValues != null) 8 * classCountersValues.length else 0 } +
      { if (attributeNormalsDescriptor != null) (for (and <- attributeNormalsDescriptor) yield and.getSize).sum else 0 }
      { if (dropped != null) 4 * dropped.length else 0 }
  }

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