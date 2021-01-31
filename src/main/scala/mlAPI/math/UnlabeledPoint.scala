package mlAPI.math

/**
 * A data point without a label. Could be used for
 * prediction or unsupervised machine learning.
 *
 * @param numericVector The numeric features.
 * @param discreteVector The discrete features.
 * @param categoricalVector The categorical features.
 */
case class UnlabeledPoint(var numericVector: Vector,
                          var discreteVector: Vector,
                          var categoricalVector: Array[String],
                          var dataInstance: String)
  extends LearningPoint {

  def this() = this(DenseVector(), DenseVector(), Array[String](), null)

  def this(numericVector: Vector) = this(numericVector, DenseVector(), Array[String](), null)

  def this(numericVector: Vector, categoricalVector: Array[String]) =
    this(numericVector, DenseVector(), categoricalVector, null)

  require(validDiscreteVector)

  override def equals(obj: Any): Boolean = {
    obj match {
      case unlabeledPoint: UnlabeledPoint =>
        numericVector.equals(unlabeledPoint.numericVector) &&
          discreteVector.equals(unlabeledPoint.discreteVector) &&
          categoricalVector.equals(unlabeledPoint.categoricalVector) &&
          dataInstance.equals(unlabeledPoint.dataInstance)
      case _ => false
    }
  }

  override def toString: String =
    s"UnlabeledPoint($numericVector, $discreteVector, ${categoricalVector.mkString("Array(", ", ", ")")})"

  override def asUnlabeledPoint: UnlabeledPoint = this

  def asLabeledPoint(label: Double): LabeledPoint =
    LabeledPoint(label, numericVector, discreteVector, categoricalVector, dataInstance)

  override def asTrainingPoint: TrainingPoint = TrainingPoint(this)
}
