package mlAPI.parameters

import breeze.linalg.{DenseVector => BreezeDenseVector, SparseVector => BreezeSparseVector}
import mlAPI.math.{DenseVector, SparseVector}

import scala.collection.mutable.ListBuffer

/** This class represents a weight vector.
 *
 * @param vector The vector of parameters.
 */
case class EuclideanVector(var vector: BreezeDenseVector[Double]) extends BreezeParameters {

  size = vector.length
  bytes = getSize * 8

  def this() = this(BreezeDenseVector.zeros[Double](1))

  def this(weights: Array[Double]) = this(BreezeDenseVector(weights.slice(0, weights.length - 1)))

  def this(denseVector: DenseVector) = this(denseVector.data)

  def this(sparseVector: SparseVector) = this(sparseVector.toDenseVector)

  def this(breezeSparseVector: BreezeSparseVector[Double]) = this(breezeSparseVector.toDenseVector)

  override def getSizes: Array[Int] = Array(vector.size)

  override def equals(obj: Any): Boolean = {
    obj match {
      case EuclideanVector(v) => vector.equals(v)
      case _ => false
    }
  }

  override def toString: String = s"EuclideanVector($vector)"

  override def +(num: Double): LearningParameters = EuclideanVector(vector + num)

  override def +=(num: Double): LearningParameters = {
    vector += num
    this
  }

  override def +(params: LearningParameters): LearningParameters = {
    params match {
      case EuclideanVector(v) => EuclideanVector(vector + v)
      case _ => throw new RuntimeException("The provided LearningParameter Object is non-compatible " +
        "for addition with a EuclideanVector Object.")
    }
  }

  override def +=(params: LearningParameters): LearningParameters = {
    params match {
      case EuclideanVector(v) =>
        vector += v
        this
      case _ => throw new RuntimeException("The provided LearningParameter Object is non-compatible " +
        "for addition with a EuclideanVector Object.")
    }
  }

  override def -(num: Double): LearningParameters = this + (-num)

  override def -=(num: Double): LearningParameters = this += (-num)

  override def -(params: LearningParameters): LearningParameters = {
    params match {
      case EuclideanVector(v) => this + EuclideanVector(-v)
      case _ => throw new RuntimeException("The provided LearningParameter Object is non-compatible " +
        "for subtraction with a EuclideanVector Object.")
    }
  }

  override def -=(params: LearningParameters): LearningParameters = {
    params match {
      case EuclideanVector(v) => this += EuclideanVector(-v)
      case _ => throw new RuntimeException("The provided LearningParameter Object is non-compatible " +
        "for subtraction with a EuclideanVector Object.")
    }
  }

  override def *(num: Double): LearningParameters = EuclideanVector(vector * num)

  override def *=(num: Double): LearningParameters = {
    vector *= num
    this
  }

  override def /(num: Double): LearningParameters = this * (1.0 / num)

  override def /=(num: Double): LearningParameters = this *= (1.0 / num)

  override def getCopy: LearningParameters = this.copy()

  override def flatten: BreezeDenseVector[Double] = vector

  override def generateSerializedParams: (LearningParameters, Array[_]) => java.io.Serializable = {
    (lPar: LearningParameters, par: Array[_]) =>
      try {
        assert(par.length == 2 && lPar.isInstanceOf[EuclideanVector])
        val sparse: Boolean = par.head.asInstanceOf[Boolean]
        val bucket: Bucket = par.tail.head.asInstanceOf[Bucket]
        (
          Array(lPar.asInstanceOf[EuclideanVector].vector.length),
          lPar.asInstanceOf[EuclideanVector].slice(bucket, sparse)
        )
      } catch {
        case _: Throwable =>
          throw new RuntimeException("Something happened while Serializing the EuclideanVector learning parameters.")
      }
  }

  override def generateParameters(pDesc: ParameterDescriptor): LearningParameters = {
    require(pDesc.getParamSizes.length == 1)
    require(pDesc.getParams.isInstanceOf[EuclideanVector])

    val weightArrays: ListBuffer[Array[Double]] =
      unwrapData(pDesc.getParamSizes, pDesc.getParams.asInstanceOf[DenseVector].data)
    assert(weightArrays.size == 1)

    EuclideanVector(BreezeDenseVector[Double](weightArrays.head))
  }
}
