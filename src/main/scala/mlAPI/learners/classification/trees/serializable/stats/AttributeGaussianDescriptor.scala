package mlAPI.learners.classification.trees.serializable.stats

import ControlAPI.CountableSerial
import com.fasterxml.jackson.core.JsonProcessingException
import com.fasterxml.jackson.databind.ObjectMapper

/**
 * A serializable descriptor of an attribute gaussian approximation.
 */
class AttributeGaussianDescriptor(var attribute: Int,
                                  var range: RangeDescriptor,
                                  var targets: Array[Int],
                                  var normals: Array[GaussianDescriptor])
  extends CountableSerial {

  override def getSize: Int = {
    4 +
      { if (range != null) range.getSize else 0 } +
      { if (targets != null) 4 * targets.length else 0 } +
      { if (normals != null) (for(normal <- normals) yield normal.getSize).sum else 0 }
  }

  def setAttribute(attribute: Int): Unit = this.attribute = attribute

  def getAttribute: Int = attribute

  def setRange(range: RangeDescriptor): Unit = this.range = range

  def getRange: RangeDescriptor = range

  def setTargets(targets: Array[Int]): Unit = this.targets = targets

  def getTargets: Array[Int] = targets

  def setNormals(normals: Array[GaussianDescriptor]): Unit = this.normals = normals

  def getNormals: Array[GaussianDescriptor] = normals

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
