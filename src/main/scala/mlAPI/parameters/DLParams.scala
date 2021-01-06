package mlAPI.parameters

import mlAPI.math.{DenseVector, Vector}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

/**
 * The learnable parameters of a Deeplearning4j neural network.
 *
 * @param parameters The [[INDArray]] parameters of the neural network.
 */
case class DLParams(var parameters: INDArray) extends VectoredParameters {

  def this() = this(null)

  size = if (parameters == null) 0 else parameters.length.toInt
  bytes = getSize * 8

  def slice(range: Bucket, sparse: Boolean): Vector = {
    sliceRequirements(range)
    val flatArray: Array[Double] = parameters.toDoubleVector.slice(range.getStart.toInt, range.getEnd.toInt + 1)
    if (sparse)
      DenseVector(flatArray).toSparseVector
    else
      DenseVector(flatArray)
  }

  override def getSizes: Array[Int] = parameters.shape().map(x => x.toInt)

  override def +(num: Double): LearningParameters = DLParams(parameters.add(num))

  override def +=(num: Double): LearningParameters = {
    parameters = parameters.add(num)
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
        parameters = parameters.add(p)
        this
      case _ => throw new RuntimeException("The provided LearningParameter Object is non-compatible " +
        "for addition with a DLParams Object.")
    }
  }

  override def -(num: Double): LearningParameters = this + (-num)

  override def -=(num: Double): LearningParameters = this += (-num)

  override def -(params: LearningParameters): LearningParameters = {
    params match {
      case DLParams(p) => this + DLParams(p.mul(-1.0))
      case _ => throw new RuntimeException("The provided LearningParameter Object is non-compatible " +
        "for subtraction with a DLParams Object.")
    }
  }

  override def -=(params: LearningParameters): LearningParameters = {
    params match {
      case DLParams(p) => this += DLParams(p.mul(-1.0))
      case _ => throw new RuntimeException("The provided LearningParameter Object is non-compatible " +
        "for subtraction with a DLParams Object.")
    }
  }

  override def *(num: Double): LearningParameters = DLParams(parameters.mul(num))

  override def *=(num: Double): LearningParameters = {
    parameters = parameters.mul(num)
    this
  }

  override def /(num: Double): LearningParameters = DLParams(parameters.div(num))

  override def /=(num: Double): LearningParameters = {
    parameters = parameters.div(num)
    this
  }

  override def FrobeniusNorm: Double = parameters.norm2Number().doubleValue()

  override def toDenseVector: Vector = DenseVector(parameters.toDoubleVector)

  override def toSparseVector: Vector = DenseVector(parameters.toDoubleVector).toSparseVector

  override def getCopy: LearningParameters = DLParams(parameters.dup())

  override def generateSerializedParams: (LearningParameters, Array[_]) => SerializedParameters = {
    (lPar: LearningParameters, par: Array[_]) =>
      try {
        assert(par.length == 2 && lPar.isInstanceOf[DLParams])
        val sparse: Boolean = par.head.asInstanceOf[Boolean]
        val bucket: Bucket = par.tail.head.asInstanceOf[Bucket]
        new SerializedVectoredParameters(
          Array(lPar.asInstanceOf[DLParams].parameters.length().toInt),
          lPar.asInstanceOf[DLParams].slice(bucket, sparse)
        )
      } catch {
        case _: Throwable =>
          throw new RuntimeException("Something happened while Serializing the DLParams learning parameters.")
      }
  }

  def setParameters(parameters: INDArray): Unit = {
    this.parameters = parameters
    size = parameters.length.toInt
    bytes = getSize * 8
  }

  override def generateParameters(pDesc: ParameterDescriptor): LearningParameters =
    DLParams(Nd4j.create(toDense(pDesc.getParams).data, pDesc.getParamSizes, 'c'))

  override def equals(obj: Any): Boolean = {
    obj match {
      case DLParams(params) => parameters.equalsWithEps(params, 0)
      case _ => false
    }
  }

}
