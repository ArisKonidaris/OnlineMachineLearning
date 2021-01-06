package mlAPI.learners.classification.trees.serializable.stats

import ControlAPI.CountableSerial
import com.fasterxml.jackson.core.JsonProcessingException
import com.fasterxml.jackson.databind.ObjectMapper

/**
 * A serializable descriptor for discrete attributes.
 */
class DiscreteAttributeDescriptor(var attributes: Array[Int],
                                  var counters: Array[ValuesDescriptor])
  extends CountableSerial{

  def setAttributes(attributes: Array[Int]): Unit = this.attributes = attributes

  def getAttributes: Array[Int] = attributes

  def setCounters(counters: Array[ValuesDescriptor]): Unit = this.counters = counters

  def getCounters: Array[ValuesDescriptor] = counters

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
    { if (attributes != null) 4 * attributes.length else 0 } +
      { if (counters != null) (for (counter <- counters) yield counter.getSize).sum else 0 }
  }

}