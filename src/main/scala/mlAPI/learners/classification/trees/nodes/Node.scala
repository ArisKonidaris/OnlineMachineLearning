package mlAPI.learners.classification.trees.nodes

import mlAPI.learners.classification.trees.serializable.nodes.NodeDescriptor
import mlAPI.math.Vector

import scala.collection.mutable

/**
 * The basic trait for a Node in a Hoeffding Tree.
 */
trait Node {

  var height: Int

  def createLeafMap(leafMap: mutable.Map[String, LeafNode]): Unit

  def predict(point: Vector, method: String = "MajorityVote"): (Int, Double)

  def filterNode(point: Vector): LeafNode

  def getNodeSize: Int

  def getSize: Int = 8

  def deactivate(): Unit

  def activate(): Unit

  def serialize: NodeDescriptor

  def generateNode: Node

}
