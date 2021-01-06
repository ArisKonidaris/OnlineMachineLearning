package mlAPI.learners.classification.trees.serializable.stats

import ControlAPI.CountableSerial
import com.fasterxml.jackson.core.JsonProcessingException
import com.fasterxml.jackson.databind.ObjectMapper

/**
 * A serializable descriptor of class distributions for each value of a discrete attribute.
 */
class ValuesDescriptor(var values: Array[Int],
                       var targetCounters: Array[TargetCountersDescriptor])
  extends CountableSerial {

  def setValues(values: Array[Int]): Unit = this.values = values

  def getValues: Array[Int] = values

  def setTargetCounters(targetCounters: Array[TargetCountersDescriptor]): Unit = this.targetCounters = targetCounters

  def getTargetCounters: Array[TargetCountersDescriptor] = targetCounters

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
    { if (values != null) 4 * values.length else 0 } +
      { if (targetCounters != null) (for (tc <- targetCounters) yield tc.getSize).sum else 0}
  }

}
