package mlAPI.learners.classification.trees.serializable.stats

import ControlAPI.CountableSerial
import com.fasterxml.jackson.core.JsonProcessingException
import com.fasterxml.jackson.databind.ObjectMapper

/**
 * A serializable descriptor of an attribute range.
 */
class RangeDescriptor(var leftEnd: Double, var rightEnd: Double) extends CountableSerial {

  def setLeftEnd(leftEnd: Double): Unit = this.leftEnd = leftEnd

  def getLeftEnd: Double = leftEnd

  def setRightEnd(rightEnd: Double): Unit = this.rightEnd = rightEnd

  def getRightEnd: Double = rightEnd

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

  override def getSize: Int = 16

}