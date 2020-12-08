package mlAPI.learners.classification.trees.serializable.stats

import com.fasterxml.jackson.core.JsonProcessingException
import com.fasterxml.jackson.databind.ObjectMapper

/**
 * A serializable descriptor of class distributions for each value of a discrete attribute.
 */
class ValuesDescriptor(var values: Array[Int],
                       var targetCounters: Array[TargetCountersDescriptor])
  extends java.io.Serializable {

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

}
