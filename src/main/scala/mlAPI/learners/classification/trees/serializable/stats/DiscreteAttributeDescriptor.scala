package mlAPI.learners.classification.trees.serializable.stats

import com.fasterxml.jackson.core.JsonProcessingException
import com.fasterxml.jackson.databind.ObjectMapper

/**
 * A serializable descriptor for discrete attributes.
 */
class DiscreteAttributeDescriptor(var attributes: Array[Int],
                                  var counters: Array[ValuesDescriptor])
  extends java.io.Serializable {

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

}