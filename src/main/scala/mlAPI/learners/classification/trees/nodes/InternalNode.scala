package mlAPI.learners.classification.trees.nodes

import mlAPI.learners.classification.trees.serializable.nodes.{InternalNodeDescriptor, LeafNodeDescriptor, NodeDescriptor}
import mlAPI.math.Vector

import scala.collection.mutable

/**
 * A class representing an internal node in a Hoeffding tree.
 *
 * @param leftChild  The left child [[Node]] of this [[InternalNode]].
 * @param rightChild The right child [[Node]] of this [[InternalNode]].
 * @param test       An attribute test imposed to a data point by this internal node to direct it to the appropriate
 *                   child [[Node]].
 * @param height     The height of the Hoeffding Tree that this internal node is on.
 */
case class InternalNode(var leftChild: Node,
                        var rightChild: Node,
                        var test: TestAttribute,
                        override var height: Int)
  extends Node {

  def this(test: TestAttribute, height: Int) = this(null, null, test, height)

  //////////////////////////////////////////////////// Getters /////////////////////////////////////////////////////////

  override def getSize: Int = getNodeSize + leftChild.getSize + rightChild.getSize

  override def getNodeSize: Int = test.getSize + 4

  //////////////////////////////////////////////////// Methods /////////////////////////////////////////////////////////

  override def filterNode(point: Vector): LeafNode = {
    if (test.test(point) < 0)
      leftChild.filterNode(point)
    else
      rightChild.filterNode(point)
  }

  override def predict(point: Vector, method: String = "MajorityVote"): (Int, Double) = {
    if (test.test(point) < 0)
      leftChild.predict(point, method)
    else
      rightChild.predict(point, method)
  }

  override def deactivate(): Unit = {}

  override def activate(): Unit = {}

  override def serialize: NodeDescriptor = {
    InternalNodeDescriptor(
      leftChild.serialize,
      rightChild.serialize,
      test.serialize,
      height
    )
  }

  override def toString: String = {
    val tabs = if (height > 1) (for (_ <- 1 until height) yield "\t").reduce(_ + _) else ""
    tabs + "If attribute_" + test.id + " <= " + test.value + " :\n" + leftChild.toString + "\n" +
      tabs + "If attribute_" + test.id + " > " + test.value + " : \n" + rightChild.toString
  }

  override def createLeafMap(leafMap: mutable.Map[String, LeafNode]): Unit = {
    leftChild.createLeafMap(leafMap)
    rightChild.createLeafMap(leafMap)
  }

  override def generateNode: Node = InternalNode(null, null, test.copy(), height)

}

object InternalNode {

  def deserialize(descriptor: InternalNodeDescriptor): InternalNode = {
    val intNode = new InternalNode(
      TestAttribute.deserialize(descriptor.getTestDescriptor),
      descriptor.getHeight
    )
    intNode.leftChild = {
      descriptor.getLeftChild match {
        case ind: InternalNodeDescriptor => InternalNode.deserialize(ind)
        case lnd: LeafNodeDescriptor =>
          val lc = LeafNode.deserialize(lnd)
          lc.parent = intNode
          lc
        case _ => throw new RuntimeException("Unknown Node Serialization scheme.")
      }
    }
    intNode.rightChild = {
      descriptor.getRightChild match {
        case ind: InternalNodeDescriptor => InternalNode.deserialize(ind)
        case lnd: LeafNodeDescriptor =>
          val rc = LeafNode.deserialize(lnd)
          rc.parent = intNode
          rc
        case _ => throw new RuntimeException("Unknown Node Serialization scheme.")
      }
    }
    intNode
  }

}
