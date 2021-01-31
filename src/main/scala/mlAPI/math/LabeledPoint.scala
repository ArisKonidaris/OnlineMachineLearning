package mlAPI.math

/**
 * This class represents a vector with an associated label as
 * it is required for many supervised learning tasks.
 *
 * @param label  Label of the data point.
 * @param numericVector  The numeric features.
 * @param discreteVector The discrete features.
 * @param categoricalVector The categorical features.
 */
case class LabeledPoint(var label: Double,
                        var numericVector: Vector,
                        var discreteVector: Vector,
                        var categoricalVector: Array[String],
                        var dataInstance: String)
  extends LearningPoint {

  def this() = this(0.0, DenseVector(), DenseVector(), Array[String](), null)
  def this(label: Double) = this(label, DenseVector(), DenseVector(), Array[String](), null)
  def this(label: Double, numericVector: Vector) = this(label, numericVector, DenseVector(), Array[String](), null)
  def this(label: Double, numericVector: Vector, categoricalVector: Array[String]) =
    this(label, numericVector, DenseVector(), categoricalVector, null)

  require(validDiscreteVector)

  def getLabel: Double = label

  def setLabel(label: Double): Unit = this.label = label

  override def equals(obj: Any): Boolean = {
    obj match {
      case labeledPoint: LabeledPoint =>
        label.equals(labeledPoint.label) &&
          numericVector.equals(labeledPoint.numericVector) &&
          discreteVector.equals(labeledPoint.discreteVector) &&
          categoricalVector.equals(labeledPoint.categoricalVector) &&
          dataInstance.equals(labeledPoint.dataInstance)
      case _ => false
    }
  }

  override def toString: String =
    s"LabeledPoint($label, $numericVector, $discreteVector, ${categoricalVector.mkString("Array(", ", ", ")")})"

  override def asTrainingPoint: TrainingPoint = TrainingPoint(this)
}
