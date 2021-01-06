package mlAPI.learners.classification.trees.serializable.stats

import ControlAPI.CountableSerial
import com.fasterxml.jackson.core.JsonProcessingException
import com.fasterxml.jackson.databind.ObjectMapper

/**
 * A serializable descriptor for a discrete statistics instance.
 */
class DiscreteStatisticsDescriptor(classCountersKeys: Array[Int],
                                   classCountersValues: Array[Double],
                                   max: Double,
                                   prediction: Int,
                                   n_l: Double,
                                   isActive: Boolean,
                                   var stats: DiscreteAttributeDescriptor,
                                   var targets: Array[Int],
                                   var dropped: Array[Int])
  extends StatisticsDescriptor(classCountersKeys, classCountersValues, max, prediction, n_l, isActive)
    with CountableSerial {

  def setStats(stats: DiscreteAttributeDescriptor): Unit = this.stats = stats

  def getStats: DiscreteAttributeDescriptor = stats

  def setTargets(targets: Array[Int]): Unit = this.targets = targets

  def getTargets: Array[Int] = targets

  def setDropped(dropped: Array[Int]): Unit = this.dropped = dropped

  def getDropped: Array[Int] = dropped

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

  override def getSize: Int = {
    21 +
      { if (classCountersKeys != null) 4 * classCountersKeys.length else 0 } +
      { if (classCountersValues != null) 8 * classCountersValues.length else 0 } +
      { if (stats != null) stats.getSize else 0 } +
      { if (targets != null) 4 * targets.length else 0 } +
      { if (dropped != null) 4 * dropped.length else 0 }
  }

}
