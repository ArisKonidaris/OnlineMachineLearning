package mlAPI.parameters

import mlAPI.math.{DenseVector, SparseVector}
import breeze.linalg.{DenseVector => BreezeDenseVector, SparseVector => BreezeSparseVector}
import mlAPI.parameters.utils.{Bucket, ParameterDescriptor, SerializableParameters}
import mlAPI.parameters.utils.{WrappedVectoredParameters => wrappedParams}

import scala.collection.mutable.ListBuffer

/** This class wraps a list of vectors with biases.
  *
  * @param vectorBiases The list of [[VectorBias]] parameter objects.
  */
case class VectorBiasList(var vectorBiases: ListBuffer[VectorBias]) extends BreezeParameters {

  size = (for (vectorBias: VectorBias <- vectorBiases) yield vectorBias.size).sum
  sizes = (for (weights: VectorBias <- vectorBiases) yield weights.getSizes).flatten.toArray
  bytes = 8 * (for (vectorBias: VectorBias <- vectorBiases) yield vectorBias.bytes).sum

  def this() = this(ListBuffer[VectorBias](new VectorBias()))

  def this(weights: Array[Double]) = this(ListBuffer[VectorBias](new VectorBias(weights)))

  def this(weights: Array[Array[Double]]) = {
    this(
      {
        val vbl: ListBuffer[VectorBias] = ListBuffer[VectorBias]()
        for (weight: Array[Double] <- weights) vbl.append(new VectorBias(weight))
        vbl
      }
    )
  }

  def this(denseVector: DenseVector) = this(denseVector.data)

  def this(denseVectors: Array[DenseVector]) = {
    this(for (denseVector: DenseVector <- denseVectors) yield denseVector.data)
  }

  def this(sparseVector: SparseVector) = this(sparseVector.toDenseVector)

  def this(sparseVectors: Array[SparseVector]) = {
    this(for (sparseVector: SparseVector <- sparseVectors) yield sparseVector.toDenseVector)
  }

  def this(breezeDenseVector: BreezeDenseVector[Double]) =
    this(ListBuffer[VectorBias](new VectorBias(breezeDenseVector)))

  def this(breezeDenseVectors: Array[BreezeDenseVector[Double]]) = {
    this(
      {
        ListBuffer[VectorBias](
          (for (breezeDenseVector: BreezeDenseVector[Double] <- breezeDenseVectors)
            yield new VectorBias(breezeDenseVector)
            ): _ *
        )
      }
    )
  }

  def this(breezeSparseVector: BreezeSparseVector[Double]) = this(breezeSparseVector.toDenseVector)

  override def equals(obj: Any): Boolean = {
    obj match {
      case VectorBiasList(vbs: ListBuffer[VectorBias]) =>
        if (vectorBiases.size != vbs.size)
          false
        else
          (
            for ((weights: VectorBias, vb: VectorBias) <- vectorBiases zip vbs)
              yield weights.equals(vb)
            ).reduce((x, y) => x && y)
      case _ => false
    }
  }

  override def toString: String = s"VectorBiasList($vectorBiases)"

  override def +(num: Double): LearningParameters = {
    VectorBiasList(for (weights: VectorBias <- vectorBiases) yield (weights + num).asInstanceOf[VectorBias])
  }

  def +(index: Int, num: Double): LearningParameters = {
    checkIndex(index)
    VectorBiasList(
      for ((weights: VectorBias, i: Int) <- vectorBiases.zipWithIndex)
        yield (if (i != index) weights else weights + num).asInstanceOf[VectorBias]
    )
  }

  override def +=(num: Double): LearningParameters = {
    for (weights: VectorBias <- vectorBiases) weights += num
    this
  }

  def +=(index: Int, num: Double): LearningParameters = {
    require(index <= vectorBiases.size - 1)
    vectorBiases(index) += num
    this
  }

  override def +(params: LearningParameters): LearningParameters = {
    params match {
      case VectorBiasList(vbs: ListBuffer[VectorBias]) =>
        VectorBiasList(
          for ((weights: VectorBias, vb: VectorBias) <- vectorBiases zip vbs)
            yield (weights + vb).asInstanceOf[VectorBias]
        )
      case vb: VectorBias =>
        VectorBiasList(for (weights: VectorBias <- vectorBiases) yield (weights + vb).asInstanceOf[VectorBias])
      case _ => throw new RuntimeException("The provided LearningParameter Object is non-compatible " +
        "for addition with a VectorBiasList Object.")
    }
  }

  def +(index: Int, params: VectorBias): LearningParameters = {
    checkIndex(index)
    VectorBiasList(
      for ((weights: VectorBias, i: Int) <- vectorBiases.zipWithIndex)
        yield (if (i != index) weights else weights + params).asInstanceOf[VectorBias]
    )
  }

  override def +=(params: LearningParameters): LearningParameters = {
    params match {
      case VectorBiasList(vbs: ListBuffer[VectorBias]) =>
        for ((weights: VectorBias, vb: VectorBias) <- vectorBiases zip vbs) weights += vb
        this
      case vb: VectorBias =>
        for (weights: VectorBias <- vectorBiases) weights += vb
        this
      case _ => throw new RuntimeException("The provided LearningParameter Object is non-compatible " +
        "for addition with a VectorBiasList Object.")
    }
  }

  def +=(index: Int, params: VectorBias): LearningParameters = {
    require(index <= vectorBiases.size - 1)
    vectorBiases(index) += params
    this
  }

  override def -(num: Double): LearningParameters = this + (-num)

  def -(index: Int, num: Double): LearningParameters = this + (index, -num)

  override def -=(num: Double): LearningParameters = this += (-num)

  def -=(index: Int, num: Double): LearningParameters = this += (index, -num)

  override def -(params: LearningParameters): LearningParameters = {
    params match {
      case VectorBiasList(vbs: ListBuffer[VectorBias]) =>
        this + VectorBiasList(for (vb <- vbs) yield (vb * -1.0).asInstanceOf[VectorBias])
      case vb: VectorBias =>
        this + (vb * -1.0)
      case _ => throw new RuntimeException("The provided LearningParameter Object is non-compatible " +
        "for subtraction with a VectorBiasList Object.")
    }
  }

  def -(index: Int, params: VectorBias): LearningParameters = this + (index, (params * -1.0).asInstanceOf[VectorBias])

  override def -=(params: LearningParameters): LearningParameters = {
    params match {
      case VectorBiasList(vbs: ListBuffer[VectorBias]) =>
        this += VectorBiasList(for (vb <- vbs) yield (vb * -1.0).asInstanceOf[VectorBias])
      case vb: VectorBias =>
        this += (vb * -1.0)
      case _ => throw new RuntimeException("The provided LearningParameter Object is non-compatible " +
        "for subtraction with a VectorBiasList Object.")
    }
  }

  def -=(index: Int, params: VectorBias): LearningParameters = this += (index, (params * -1.0).asInstanceOf[VectorBias])

  override def *(num: Double): LearningParameters =
    VectorBiasList(for (weights: VectorBias <- vectorBiases) yield (weights * num).asInstanceOf[VectorBias])

  def *(index: Int, num: Double): LearningParameters = {
    require(index <= vectorBiases.size - 1)
    VectorBiasList(for ((weights: VectorBias, i: Int) <- vectorBiases.zipWithIndex)
      yield (if (i != index) weights else weights * num).asInstanceOf[VectorBias]
    )
  }

  override def *=(num: Double): LearningParameters = {
    for (weights: VectorBias <- vectorBiases) weights *= num
    this
  }

  def *=(index: Int, num: Double): LearningParameters = {
    require(index <= vectorBiases.size - 1)
    vectorBiases(index) *= num
    this
  }

  override def /(num: Double): LearningParameters = this * (1.0 / num)

  def /(index: Int, num: Double): LearningParameters = this * (index, 1.0 / num)

  override def /=(num: Double): LearningParameters = this *= (1.0 / num)

  def /=(index: Int, num: Double): LearningParameters = this *= (index, 1.0 / num)

  override def getCopy: LearningParameters = {
    val newVectorBiases: ListBuffer[VectorBias] = ListBuffer[VectorBias]()
    for (vectorBias <- vectorBiases)
      newVectorBiases += vectorBias.getCopy.asInstanceOf[VectorBias]
    VectorBiasList(newVectorBiases)
  }

  override def flatten: BreezeDenseVector[Double] =
    (for (weights: VectorBias <- vectorBiases) yield weights.flatten).reduce((x, y) => BreezeDenseVector.vertcat(x, y))

  override def extractParams: (LearningParameters, Boolean) => SerializableParameters = {
    (params: LearningParameters, sparse : Boolean) =>
      try {
        assert(params.isInstanceOf[VectorBiasList])
        wrappedParams(
          (for (weights: VectorBias <- vectorBiases) yield weights.getSizes).flatten.toArray,
          null,
          params.asInstanceOf[VectorBiasList].slice(Bucket(0, params.asInstanceOf[VectorBiasList].size - 1), sparse)
        )
      } catch {
        case _: Throwable =>
          throw new RuntimeException("Something happened while extracting the VectorBiasList learning parameters.")
      }
  }

  override def extractDivParams: (LearningParameters, Array[_]) => Array[Array[SerializableParameters]] = {
    (params: LearningParameters, args: Array[_]) =>
      try {
        assert(params.isInstanceOf[VectorBiasList] && args.length == 2)
        val (sparse, quantiles) = extractSparseQuantiles(args)
        val wrapped = ListBuffer[Array[SerializableParameters]]()
        for (buckets: Array[Bucket] <- quantiles)
          wrapped append {
            val hubWrap = for (bucket: Bucket <- buckets)
              yield wrappedParams(null, bucket, params.asInstanceOf[VectorBiasList].slice(bucket, sparse))
            hubWrap.head.setSizes((for (weights: VectorBias <- vectorBiases) yield weights.getSizes).flatten.toArray)
            hubWrap.asInstanceOf[Array[SerializableParameters]]
          }
        wrapped.toArray
      } catch {
        case _: Throwable =>
          throw new RuntimeException("Something happened while extracting the divided VectorBiasList learning parameters.")
      }
  }

  override def generateParameters(pDesc: ParameterDescriptor): LearningParameters = {
    try {
      val weightArrays: Array[Array[Double]] = unwrapData(pDesc.getParamSizes, toDense(pDesc.getParams).data)
      new VectorBiasList(for ((e, i) <- weightArrays.zipWithIndex if i % 2 == 0) yield e ++ weightArrays(i + 1))
    } catch {
      case _: Throwable =>
        throw new RuntimeException("Something happened while deserializing the VectorBiasList learning parameters.")
    }
  }

  def checkIndex(index: Int): Unit = require(index <= vectorBiases.size - 1)

}
