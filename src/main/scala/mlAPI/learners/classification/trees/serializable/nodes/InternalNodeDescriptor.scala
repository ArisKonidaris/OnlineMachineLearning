package mlAPI.learners.classification.trees.serializable.nodes

import com.fasterxml.jackson.core.JsonProcessingException
import com.fasterxml.jackson.databind.ObjectMapper

/**
 * A serializable descriptor of an internal node.
 */
case class InternalNodeDescriptor(var leftChild: NodeDescriptor,
                                  var rightChild: NodeDescriptor,
                                  var test: TestDescriptor,
                                  var height: Int)
  extends NodeDescriptor with java.io.Serializable {

  def setLeftChild(leftChild: NodeDescriptor): Unit = this.leftChild = leftChild

  def getLeftChild: NodeDescriptor = leftChild

  def setRightChild(rightChild: NodeDescriptor): Unit = this.rightChild = rightChild

  def getRightChild: NodeDescriptor = rightChild

  def setTest(test: TestDescriptor): Unit = this.test = test

  def getTestDescriptor: TestDescriptor = test

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
