package mlAPI.math

/**
 * A trait representing a data point required for
 * machine learning tasks.
 */
trait Point extends Serializable {

  var numericVector: Vector
  var discreteVector: Vector
  var categoricalVector: Array[String]

  def validDiscreteVector: Boolean = {
    if (discreteVector.size == 0)
      true
    else
      (
        for (value: Double <- discreteVector.asInstanceOf[DenseVector].data) yield value == Math.floor(value)
        ).reduce((x,y) => x && y)
  }

  def setNumericVector(vector: Vector): Unit = this.numericVector = vector

  def getNumericVector: Vector = numericVector

  def setDiscreteVector(vector: Vector): Unit = this.discreteVector = vector

  def getDiscreteVector: Vector = discreteVector

  def setCategoricalVector(vector: Array[String]): Unit = this.categoricalVector = vector

  def getCategoricalVector: Array[String] = categoricalVector

  def numericToList: List[Double] = numericVector.toList

  def discreteToList: List[Double] = discreteVector.toList

  def categoricalToList: List[String] = categoricalVector.toList

  def marshal(): (Array[Int], Array[Double], Array[String]) = {
    (
      Array[Int](numericVector.size, discreteVector.size),
      {
        val ar1: Array[Double] = numericVector.toList.toArray
        val ar2: Array[Double] = discreteVector.toList.toArray
        ar1 ++ ar2
      },
      categoricalVector
    )
  }

}

/** A data point without a label. Could be used for
 * prediction or unsupervised machine learning.
 *
 * @param numericVector  The numeric features.
 * @param discreteVector The discrete features.
 * @param categoricalVector The categorical features.
 */
case class UnlabeledPoint(var numericVector: Vector, var discreteVector: Vector, var categoricalVector: Array[String])
  extends Point {

  def this() = this(DenseVector(), DenseVector(), Array[String]())

  def this(numericVector: Vector) = this(numericVector, DenseVector(), Array[String]())

  def this(numericVector: Vector, categoricalVector: Array[String]) =
    this(numericVector, DenseVector(), categoricalVector)

  require(validDiscreteVector)

  override def equals(obj: Any): Boolean = {
    obj match {
      case unlabeledPoint: UnlabeledPoint =>
        numericVector.equals(unlabeledPoint.numericVector) &&
          discreteVector.equals(unlabeledPoint.discreteVector) &&
          categoricalVector.equals(unlabeledPoint.categoricalVector)
      case _ => false
    }
  }

  override def toString: String =
    s"UnlabeledPoint($numericVector, $discreteVector, ${categoricalVector.mkString("Array(", ", ", ")")})"

  def convertToLabeledPoint(label: Double): LabeledPoint =
    LabeledPoint(label, numericVector, discreteVector, categoricalVector)

}

/** This class represents a vector with an associated label as it is
 * required for many supervised learning tasks.
 *
 * @param label  Label of the data point.
 * @param numericVector  The numeric features.
 * @param discreteVector The discrete features.
 * @param categoricalVector The categorical features.
 */
case class LabeledPoint(var label: Double,
                        var numericVector: Vector,
                        var discreteVector: Vector,
                        var categoricalVector: Array[String])
  extends Point {

  def this() = this(0.0, DenseVector(), DenseVector(), Array[String]())
  def this(label: Double) = this(label, DenseVector(), DenseVector(), Array[String]())
  def this(label: Double, numericVector: Vector) = this(label, numericVector, DenseVector(), Array[String]())
  def this(label: Double, numericVector: Vector, categoricalVector: Array[String]) =
    this(label, numericVector, DenseVector(), categoricalVector)

  require(validDiscreteVector)

  def getLabel: Double = label

  def setLabel(label: Double): Unit = this.label = label

  override def equals(obj: Any): Boolean = {
    obj match {
      case labeledPoint: LabeledPoint =>
        label.equals(labeledPoint.label) &&
          numericVector.equals(labeledPoint.numericVector) &&
          discreteVector.equals(labeledPoint.discreteVector) &&
          categoricalVector.equals(labeledPoint.categoricalVector)
      case _ => false
    }
  }

  override def toString: String =
    s"LabeledPoint($label, $numericVector, $discreteVector, ${categoricalVector.mkString("Array(", ", ", ")")})"

  def convertToUnlabeledPoint(): UnlabeledPoint = UnlabeledPoint(numericVector, discreteVector, categoricalVector)

}
