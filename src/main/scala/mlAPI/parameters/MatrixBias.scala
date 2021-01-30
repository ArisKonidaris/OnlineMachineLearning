package mlAPI.parameters

import breeze.linalg.{DenseMatrix => BreezeDenseMatrix, DenseVector => BreezeDenseVector}
import mlAPI.parameters.utils.{Bucket, ParameterDescriptor, SerializableParameters}
import mlAPI.parameters.utils.{WrappedVectoredParameters => wrappedParams}

import scala.collection.mutable.ListBuffer

/** This class represents a weight matrix with an intercept (bias) vector.
 *
 * @param A The matrix of parameters.
 * @param b The intercept (bias) vector weight.
 */
case class MatrixBias(var A: BreezeDenseMatrix[Double], var b: BreezeDenseVector[Double])
  extends BreezeParameters {

  size = A.cols * A.rows + b.length
  sizes = Array(A.size, b.size)
  bytes = 8 * size

  def this() = this(BreezeDenseMatrix.zeros(1, 1), BreezeDenseVector.zeros(1))

  override def equals(obj: Any): Boolean = {
    obj match {
      case MatrixBias(w, i) => b == i && A.equals(w)
      case _ => false
    }
  }

  override def toString: String = s"MatrixBias([${A.rows}x${A.cols}], ${A.toDenseVector}, $b)"

  override def +(num: Double): LearningParameters = MatrixBias(A + num, b + num)

  override def +=(num: Double): LearningParameters = {
    A = A + num
    b = b + num
    this
  }

  override def +(params: LearningParameters): LearningParameters = {
    params match {
      case MatrixBias(a, b_) => MatrixBias(A + a, b + b_)
      case _ => throw new RuntimeException("The provided LearningParameter Object is non-compatible " +
        "for addition with a MatrixBias Object.")
    }
  }

  override def +=(params: LearningParameters): LearningParameters = {
    params match {
      case MatrixBias(a, _b) =>
        A = A + a
        b = b + _b
        this
      case _ => throw new RuntimeException("The provided LearningParameter Object is non-compatible " +
        "for addition with a MatrixBias Object.")
    }
  }

  override def -(num: Double): LearningParameters = this + (-num)

  override def -=(num: Double): LearningParameters = this += (-num)

  override def -(params: LearningParameters): LearningParameters = {
    params match {
      case MatrixBias(a, b_) => this + MatrixBias(-a, -b_)
      case _ => throw new RuntimeException("The provided LearningParameter Object is non-compatible " +
        "for subtraction with a MatrixBias Object.")
    }
  }

  override def -=(params: LearningParameters): LearningParameters = {
    params match {
      case MatrixBias(a, b_) => this += MatrixBias(-a, -b_)
      case _ => throw new RuntimeException("The provided LearningParameter Object is non-compatible " +
        "for subtraction with a MatrixBias Object.")
    }
  }

  override def *(num: Double): LearningParameters = MatrixBias(A * num, b * num)

  override def *=(num: Double): LearningParameters = {
    A = A * num
    b = b * num
    this
  }

  override def /(num: Double): LearningParameters = this * (1.0 / num)

  override def /=(num: Double): LearningParameters = this *= (1.0 / num)

  override def getCopy: LearningParameters = {
    val A_ = A.copy
    val b_ = b.copy
    MatrixBias(A_, b_)
  }

  override def flatten: BreezeDenseVector[Double] = BreezeDenseVector.vertcat(A.toDenseVector, b)

  override def extractParams: (LearningParameters, Boolean) => SerializableParameters = {
    (params: LearningParameters, sparse: Boolean) =>
      try {
        assert(params.isInstanceOf[MatrixBias])
        val bucket: Bucket = Bucket(0, params.asInstanceOf[MatrixBias].size - 1)
        wrappedParams(
          Array(params.asInstanceOf[MatrixBias].A.size, params.asInstanceOf[MatrixBias].b.size),
          null,
          params.asInstanceOf[MatrixBias].slice(bucket, sparse)
        )
      } catch {
        case _: Throwable =>
          throw new RuntimeException("Something happened while extracting the MatrixBias learning parameters.")
      }
  }

  override def extractDivParams: (LearningParameters, Array[_]) => Array[Array[SerializableParameters]] = {
    (params: LearningParameters, args: Array[_]) =>
      try {
        assert(params.isInstanceOf[MatrixBias] && args.length == 2)
        val (sparse, quantiles) = extractSparseQuantiles(args)
        val wrapped = ListBuffer[Array[SerializableParameters]]()
        for (buckets: Array[Bucket] <- quantiles)
          wrapped append {
            val hubWrap = for (bucket: Bucket <- buckets)
              yield wrappedParams(null, bucket, params.asInstanceOf[MatrixBias].slice(bucket, sparse))
            hubWrap.head.setSizes(Array(params.asInstanceOf[MatrixBias].A.size, params.asInstanceOf[MatrixBias].b.size))
            hubWrap.asInstanceOf[Array[SerializableParameters]]
          }
        wrapped.toArray
      } catch {
        case _: Throwable =>
          throw new RuntimeException("Something happened while extracting the divided MatrixBias learning parameters.")
      }
  }

  override def generateParameters(pDesc: ParameterDescriptor): LearningParameters = {
    try {
      val weightArrays: Array[Array[Double]] = unwrapData(pDesc.getParamSizes, toDense(pDesc.getParams).data)
      MatrixBias(BreezeDenseVector[Double](weightArrays.head).toDenseMatrix,
        BreezeDenseVector[Double](weightArrays.tail.head))
    } catch {
      case _: Throwable =>
        throw new RuntimeException("Something happened while deserializing the MatrixBias learning parameters.")
    }
  }

}

