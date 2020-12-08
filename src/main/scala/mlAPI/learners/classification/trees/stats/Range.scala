package mlAPI.learners.classification.trees.stats

import mlAPI.learners.classification.trees.serializable.stats.RangeDescriptor

/**
 * The value range of an attribute.
 *
 * @param leftEnd  The minimum value of the attribute.
 * @param rightEnd The maximum value of the attribute.
 */
case class Range(private var leftEnd: Double,
                 private var rightEnd: Double) {

  def this() = this(Double.MaxValue, Double.MinValue)

  def getRange: Double = rightEnd - leftEnd

  def getLeftEnd: Double = leftEnd

  def getRightEnd: Double = rightEnd

  def setLeftEnd(min: Double): Unit = this.leftEnd = min

  def setRightEnd(max: Double): Unit = this.rightEnd = max

  def update(value: Double): Unit = {
    if (value < leftEnd) leftEnd = value
    if (value > rightEnd) rightEnd = value
  }

  def serialize: RangeDescriptor = new RangeDescriptor(leftEnd, rightEnd)

}

object Range {

  def getSize: Int = 16

  def deserialize(descriptor: RangeDescriptor): Range = new Range(descriptor.getLeftEnd, descriptor.getRightEnd)

}
