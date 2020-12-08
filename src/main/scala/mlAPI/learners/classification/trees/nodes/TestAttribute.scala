package mlAPI.learners.classification.trees.nodes

import mlAPI.math.Vector
import mlAPI.learners.classification.trees.serializable.nodes.TestDescriptor

/**
 * This is a class containing a test to a specific attribute. [[TestAttribute]] instances are
 * used by the internal nodes of the Hoeffding tree to direct a data point to the appropriate child [[Node]].
 *
 * @param id    The id of the attribute.
 * @param value The decision value of the attribute.
 */
case class TestAttribute(var id: Int = 0, var value: Double = 0.0) {

  require(id >= 0)

  def test(vector: Vector): Int = {
    require(id < vector.size)
    if (vector(id) <= value) -1 else 1
  }

  def getSize: Int = 12

  def serialize: TestDescriptor = new TestDescriptor(id, value)

}

object TestAttribute {
  def deserialize(descriptor: TestDescriptor): TestAttribute = new TestAttribute(descriptor.getId, descriptor.getValue)
}
