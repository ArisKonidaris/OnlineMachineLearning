package mlAPI.learners.classification.trees.serializable.stats

import ControlAPI.CountableSerial
import com.fasterxml.jackson.core.JsonProcessingException
import com.fasterxml.jackson.databind.ObjectMapper

/**
 * A serializable descriptor for a class distribution.
 */
class TargetCountersDescriptor(var targets: Array[Int], var counters: Array[Long]) extends CountableSerial {

  def setTargets(targets: Array[Int]): Unit = this.targets = targets

  def getTargets: Array[Int] = targets

  def setCounters(counters: Array[Long]): Unit = this.counters = counters

  def getCounters: Array[Long] = counters

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
    { if (targets != null) 4 * targets.length else 0 } +
      { if (counters != null) 8 * counters.length else 0 }
  }

}
