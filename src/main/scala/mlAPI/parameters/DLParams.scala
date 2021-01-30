package mlAPI.parameters

import mlAPI.math.{DenseVector, Vector}
import mlAPI.parameters.utils.{Bucket, ParameterDescriptor, SerializableParameters}
import mlAPI.parameters.utils.{WrappedVectoredParameters => wrappedParams}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.collection.mutable.ListBuffer

/**
 * The learnable parameters of a Deeplearning4j neural network.
 *
 * @param parameters The [[INDArray]] parameters of the neural network.
 */
case class DLParams(var parameters: INDArray) extends VectoredParameters {

  def this() = this(null)

  size = if (parameters == null) 0 else parameters.length.toInt
  sizes = if (parameters == null) null else parameters.shape().map(x => x.toInt)
  bytes = getSize * 8

  def getParameters: INDArray = parameters

  def setParameters(parameters: INDArray): Unit = {
    require(parameters != null)
    this.parameters = parameters
    size = parameters.length.toInt
    sizes = parameters.shape().map(x => x.toInt)
    bytes = getSize * 8
  }

  // TODO: WHATS THIS MOTHERFUCKER.
  def slice(range: Bucket, sparse: Boolean): Vector = {
    require(range.getEnd <= getSize)
    val flatArray: Array[Double] = {
      if (range.getLength == size)
        parameters.toDoubleVector
      else
        parameters.toDoubleVector.slice(range.getStart.toInt, range.getEnd.toInt + 1)
    }
    if (sparse)
      DenseVector(flatArray).toSparseVector
    else
      DenseVector(flatArray)
  }

  override def +(num: Double): LearningParameters = DLParams(parameters.add(num))

  override def +=(num: Double): LearningParameters = {
    parameters.addi(num)
    this
  }

  override def +(params: LearningParameters): LearningParameters = {
    params match {
      case DLParams(p) => DLParams(parameters.add(p))
      case _ => throw new RuntimeException("The provided LearningParameter Object is non-compatible " +
        "for addition with a DLParams Object.")
    }
  }

  override def +=(params: LearningParameters): LearningParameters = {
    params match {
      case DLParams(p) =>
        parameters.addi(p)
        this
      case _ => throw new RuntimeException("The provided LearningParameter Object is non-compatible " +
        "for addition with a DLParams Object.")
    }
  }

  override def -(num: Double): LearningParameters = this + (-num)

  override def -=(num: Double): LearningParameters = this += (-num)

  override def -(params: LearningParameters): LearningParameters = {
    params match {
      case DLParams(p) => DLParams(parameters.sub(p))
      case _ => throw new RuntimeException("The provided LearningParameter Object is non-compatible " +
        "for subtraction with a DLParams Object.")
    }
  }

  override def -=(params: LearningParameters): LearningParameters = {
    params match {
      case DLParams(p) =>
        parameters.subi(p)
        this
      case _ => throw new RuntimeException("The provided LearningParameter Object is non-compatible " +
        "for subtraction with a DLParams Object.")
    }
  }

  override def *(num: Double): LearningParameters = DLParams(parameters.mul(num))

  override def *=(num: Double): LearningParameters = {
    parameters.muli(num)
    this
  }

  override def /(num: Double): LearningParameters = DLParams(parameters.div(num))

  override def /=(num: Double): LearningParameters = {
    parameters.divi(num)
    this
  }

  override def frobeniusNorm: Double = parameters.norm2Number().doubleValue()

  override def toDenseVector: Vector = DenseVector(parameters.toDoubleVector)

  override def toSparseVector: Vector = DenseVector(parameters.toDoubleVector).toSparseVector

  override def getCopy: LearningParameters = DLParams(parameters.dup())

  override def equals(obj: Any): Boolean = {
    obj match {
      case DLParams(params) => parameters.equalsWithEps(params, 0)
      case _ => false
    }
  }

  override def toString: String = s"DLParams($parameters)"

  override def extractParams: (LearningParameters, Boolean) => SerializableParameters = {
    (params: LearningParameters, sparse: Boolean) =>
      try {
        assert(params.isInstanceOf[DLParams])
        val bucket: Bucket = Bucket(0, params.asInstanceOf[DLParams].size - 1)
        wrappedParams(parameters.shape().map(x => x.toInt), null, params.asInstanceOf[DLParams].slice(bucket, sparse))
      } catch {
        case _: Throwable =>
          throw new RuntimeException("Something happened while extracting the DLParams learning parameters.")
      }
  }

  override def extractDivParams: (LearningParameters, Array[_]) => Array[Array[SerializableParameters]] = {
    (params: LearningParameters, args: Array[_]) =>
      try {
        assert(params.isInstanceOf[DLParams] && args.length == 2)
        val (sparse, quantiles) = extractSparseQuantiles(args)
        val wrapped = ListBuffer[Array[SerializableParameters]]()
        for (buckets: Array[Bucket] <- quantiles)
          wrapped append {
            val hubWrap = for (bucket: Bucket <- buckets)
              yield wrappedParams(null, bucket, params.asInstanceOf[DLParams].slice(bucket, sparse))
            hubWrap.head.setSizes(parameters.shape().map(x => x.toInt))
            hubWrap.asInstanceOf[Array[SerializableParameters]]
          }
        wrapped.toArray
      } catch {
        case _: Throwable =>
          throw new RuntimeException("Something happened while extracting the divided DLParams learning parameters.")
      }
  }

  override def generateParameters(pDesc: ParameterDescriptor): LearningParameters = {
    try {
      DLParams(Nd4j.create(toDense(pDesc.getParams).data, pDesc.getParamSizes, 'c'))
    } catch {
      case _: Throwable =>
        throw new RuntimeException("Something happened while deserializing the DLParams learning parameters.")
    }
  }

}
