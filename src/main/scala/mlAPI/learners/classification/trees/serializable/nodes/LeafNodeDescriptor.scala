package mlAPI.learners.classification.trees.serializable.nodes

import com.fasterxml.jackson.core.JsonProcessingException
import com.fasterxml.jackson.databind.ObjectMapper
import mlAPI.learners.classification.trees.serializable.stats.StatisticsDescriptor

/**
 * A serializable descriptor of a leaf node.
 */
case class LeafNodeDescriptor(var id: Int,
                              var isLeft: Boolean,
                              var stats: StatisticsDescriptor,
                              var height: Int)
  extends NodeDescriptor with java.io.Serializable {

  def setId(id: Int): Unit = this.id = id

  def getId: Int = id

  def setISLeft(isLeft: Boolean): Unit = this.isLeft = isLeft

  def getIsLeft: Boolean = isLeft

  def setStats(stats: StatisticsDescriptor): Unit = this.stats = stats

  def getStats: StatisticsDescriptor = stats

  def setHeight(height: Int): Unit = this.height = height

  def getHeight: Int = height

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
