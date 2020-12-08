package mlAPI.learners.classification.trees.serializable.stats

import com.fasterxml.jackson.core.JsonProcessingException
import com.fasterxml.jackson.databind.ObjectMapper

/**
 * A serializable descriptor of an attribute gaussian approximation.
 */
class AttributeGaussianDescriptor(var attribute: Int,
                                  var range: RangeDescriptor,
                                  var targets: Array[Int],
                                  var normals: Array[GaussianDescriptor]) extends java.io.Serializable {

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
