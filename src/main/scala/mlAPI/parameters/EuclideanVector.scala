package mlAPI.parameters

import breeze.linalg.{DenseVector => BreezeDenseVector, SparseVector => BreezeSparseVector}
import mlAPI.math.{DenseVector, SparseVector}
import mlAPI.parameters.utils.{Bucket, ParameterDescriptor, SerializableParameters}
import mlAPI.parameters.utils.{WrappedVectoredParameters => wrappedParams}

import scala.collection.mutable.ListBuffer

/** This class represents a weight vector.
 *
 * @param vector The vector of parameters.
 */
case class EuclideanVector(var vector: BreezeDenseVector[Double]) extends BreezeParameters {

  size = vector.length
  sizes = Array(vector.length)
  bytes = getSize * 8

  def this() = this(BreezeDenseVector.zeros[Double](1))

  def this(weights: Array[Double]) = this(BreezeDenseVector(weights.slice(0, weights.length - 1)))

  def this(denseVector: DenseVector) = this(denseVector.data)

  def this(sparseVector: SparseVector) = this(sparseVector.toDenseVector)

  def this(breezeSparseVector: BreezeSparseVector[Double]) = this(breezeSparseVector.toDenseVector)

  override def toString: String = s"EuclideanVector($vector)"

  override def equals(obj: Any): Boolean = {
    obj match {
      case EuclideanVector(v) => vector.equals(v)
      case _ => false
    }
  }

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

  override def getCopy: LearningParameters = {
    val v = vector.copy
    EuclideanVector(v)
  }

  override def flatten: BreezeDenseVector[Double] = vector

  override def extractParams: (LearningParameters, Boolean) => SerializableParameters = {
    (params: LearningParameters, sparse: Boolean) =>
      try {
        assert(params.isInstanceOf[EuclideanVector])
        val bucket: Bucket = Bucket(0, params.asInstanceOf[EuclideanVector].size - 1)
        wrappedParams(
          Array(params.asInstanceOf[EuclideanVector].vector.length),
          null,
          params.asInstanceOf[EuclideanVector].slice(bucket, sparse)
        )
      } catch {
        case _: Throwable =>
          throw new RuntimeException("Something happened while extracting the EuclideanVector learning parameters.")
      }
  }

  override def extractDivParams: (LearningParameters, Array[_]) => Array[Array[SerializableParameters]] = {
    (params: LearningParameters, args: Array[_]) =>
      try {
        assert(params.isInstanceOf[EuclideanVector] && args.length == 2)
        val (sparse, quantiles) = extractSparseQuantiles(args)
        val wrapped = ListBuffer[Array[SerializableParameters]]()
        for (buckets: Array[Bucket] <- quantiles)
          wrapped append {
            val hubWrap = for (bucket: Bucket <- buckets)
              yield wrappedParams(null, bucket, params.asInstanceOf[EuclideanVector].slice(bucket, sparse))
            hubWrap.head.setSizes(Array(params.asInstanceOf[EuclideanVector].vector.length))
            hubWrap.asInstanceOf[Array[SerializableParameters]]
          }
        wrapped.toArray
      } catch {
        case _: Throwable =>
          throw new RuntimeException("Something happened while extracting the divided EuclideanVector learning parameters.")
      }
  }

  override def generateParameters(pDesc: ParameterDescriptor): LearningParameters = {
    try {
      val weightArrays: Array[Array[Double]] = unwrapData(pDesc.getParamSizes, toDense(pDesc.getParams).data)
      EuclideanVector(BreezeDenseVector[Double](weightArrays.head))
    } catch {
      case _: Throwable =>
        throw new RuntimeException("Something happened while deserializing the EuclideanVector learning parameters.")
    }
  }

}
