package mlAPI.parameters

import ControlAPI.CountableSerial
import mlAPI.math.{DenseVector, SparseVector}
import breeze.linalg.{DenseVector => BreezeDenseVector, SparseVector => BreezeSparseVector}

import scala.collection.mutable.ListBuffer

/** This class represents a weight vector with an intercept (bias).
 *
 * @param weights   The vector of parameters.
 * @param intercept The intercept (bias) weight.
 */
case class VectorBias(var weights: BreezeDenseVector[Double], var intercept: Double)
  extends BreezeParameters {

  size = weights.length + 1
  bytes = getSize * 8

  def this() = this(BreezeDenseVector.zeros(1), 0)

  def this(weights: Array[Double]) = this(
    BreezeDenseVector(weights.slice(0, weights.length - 1)),
    weights(weights.length - 1)
  )

  def this(denseVector: DenseVector) = this(denseVector.data)

  def this(sparseVector: SparseVector) = this(sparseVector.toDenseVector)

  def this(breezeDenseVector: BreezeDenseVector[Double]) =
    this(breezeDenseVector(0 to breezeDenseVector.length - 2),
      breezeDenseVector.valueAt(breezeDenseVector.length - 1)
    )

  def this(breezeSparseVector: BreezeSparseVector[Double]) = this(breezeSparseVector.toDenseVector)

  override def getSizes: Array[Int] = Array(weights.size, 1)

  override def equals(obj: Any): Boolean = {
    obj match {
      case VectorBias(w, i) => intercept == i && weights.equals(w)
      case _ => false
    }
  }

  override def toString: String = s"VectorBias($weights, $intercept)"

  override def +(num: Double): LearningParameters = VectorBias(weights + num, intercept + num)

  override def +=(num: Double): LearningParameters = {
    weights += num
    intercept += num
    this
  }

  override def +(params: LearningParameters): LearningParameters = {
    params match {
      case VectorBias(w, i) => VectorBias(weights + w, intercept + i)
      case _ => throw new RuntimeException("The provided LearningParameter Object is non-compatible " +
        "for addition with a VectorBias Object.")
    }
  }

  override def +=(params: LearningParameters): LearningParameters = {
    params match {
      case VectorBias(w, i) =>
        weights += w
        intercept += i
        this
      case _ => throw new RuntimeException("The provided LearningParameter Object is non-compatible " +
        "for addition with a VectorBias Object.")
    }
  }

  override def -(num: Double): LearningParameters = this + (-num)

  override def -=(num: Double): LearningParameters = this += (-num)

  override def -(params: LearningParameters): LearningParameters = {
    params match {
      case VectorBias(w, i) => this + VectorBias(-w, -i)
      case _ => throw new RuntimeException("The provided LearningParameter Object is non-compatible " +
        "for subtraction with a VectorBias Object.")
    }
  }

  override def -=(params: LearningParameters): LearningParameters = {
    params match {
      case VectorBias(w, i) => this += VectorBias(-w, -i)
      case _ => throw new RuntimeException("The provided LearningParameter Object is non-compatible " +
        "for subtraction with a VectorBias Object.")
    }
  }

  override def *(num: Double): LearningParameters = VectorBias(weights * num, intercept * num)

  override def *=(num: Double): LearningParameters = {
    weights *= num
    intercept *= num
    this
  }

  override def /(num: Double): LearningParameters = this * (1.0 / num)

  override def /=(num: Double): LearningParameters = this *= (1.0 / num)

  override def getCopy: LearningParameters = {
    val w = weights.copy
    val i = intercept
    VectorBias(w, i)
  }

  override def flatten: BreezeDenseVector[Double] =
    BreezeDenseVector.vertcat(weights, BreezeDenseVector.fill(1) {
      intercept
    })

  override def generateSerializedParams: (LearningParameters, Array[_]) => SerializedParameters = {
    (lPar: LearningParameters, par: Array[_]) =>
      try {
        assert(par.length == 2 && lPar.isInstanceOf[VectorBias])
        val sparse: Boolean = par.head.asInstanceOf[Boolean]
        val bucket: Bucket = par.tail.head.asInstanceOf[Bucket]
        new SerializedVectoredParameters(
          Array(lPar.asInstanceOf[VectorBias].weights.length, 1),
          lPar.asInstanceOf[VectorBias].slice(bucket, sparse)
        )
      } catch {
        case _: Throwable =>
          throw new RuntimeException("Something happened while Serializing the VectorBias learning parameters.")
      }
  }

  override def generateParameters(pDesc: ParameterDescriptor): LearningParameters = {
    require(pDesc.getParamSizes.length == 2 && pDesc.getParamSizes.tail.head == 1)
    val weightArrays: ListBuffer[Array[Double]] = unwrapData(pDesc.getParamSizes, toDense(pDesc.getParams).data)
    assert(weightArrays.size == 2)
    VectorBias(BreezeDenseVector[Double](weightArrays.head), weightArrays.tail.head.head)
  }

}