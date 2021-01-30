package mlAPI.parameters

import mlAPI.math.{DenseVector, SparseVector, Vector}
import breeze.linalg.{DenseVector => BreezeDenseVector, SparseVector => BreezeSparseVector}
import mlAPI.parameters.utils.{Bucket, ParameterDescriptor, SerializableParameters}
import mlAPI.parameters.utils.{WrappedVectoredParameters => wrappedParams}

import scala.collection.mutable.ListBuffer

/** This class wraps a list of vectors.
  *
  * @param vectors The list of [[EuclideanVector]] parameter objects.
  */
case class VectorList(var vectors: ListBuffer[EuclideanVector]) extends BreezeParameters {

  size = (for (vector: EuclideanVector <- vectors) yield vector.size).sum
  sizes = (for (vector: EuclideanVector <- vectors) yield vector.getSizes).flatten.toArray
  bytes = 8 * (for (vector: EuclideanVector <- vectors) yield vector.bytes).sum

  def this() = this(ListBuffer[EuclideanVector](new EuclideanVector()))

  def this(weights: Array[Double]) = this(ListBuffer[EuclideanVector](new EuclideanVector(weights)))

  def this(weights: Array[Array[Double]]) = {
    this(
      {
        val vbl: ListBuffer[EuclideanVector] = ListBuffer[EuclideanVector]()
        for (weight: Array[Double] <- weights) vbl.append(new EuclideanVector(weight))
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
    this(ListBuffer[EuclideanVector](EuclideanVector(breezeDenseVector)))

  def this(breezeDenseVectors: Array[BreezeDenseVector[Double]]) = {
    this(
      {
        ListBuffer[EuclideanVector](
          (for (breezeDenseVector: BreezeDenseVector[Double] <- breezeDenseVectors)
            yield EuclideanVector(breezeDenseVector)
            ): _ *
        )
      }
    )
  }

  def this(breezeSparseVector: BreezeSparseVector[Double]) = this(breezeSparseVector.toDenseVector)

  override def equals(obj: Any): Boolean = {
    obj match {
      case VectorList(vs: ListBuffer[EuclideanVector]) =>
        if (vectors.size != vs.size)
          false
        else
          (
            for ((vector: EuclideanVector, v: EuclideanVector) <- vectors zip vs)
              yield vector.equals(v)
            ).reduce((x, y) => x && y)
      case _ => false
    }
  }

  override def toString: String = s"VectorList($vectors)"

  override def +(num: Double): LearningParameters = {
    VectorList(for (vector: EuclideanVector <- vectors) yield (vector + num).asInstanceOf[EuclideanVector])
  }

  def +(index: Int, num: Double): LearningParameters = {
    require(index <= vectors.size - 1)
    VectorList(
      for ((vector: EuclideanVector, i: Int) <- vectors.zipWithIndex)
        yield (if (i != index) vector else vector + num).asInstanceOf[EuclideanVector]
    )
  }

  override def +=(num: Double): LearningParameters = {
    for (vector: EuclideanVector <- vectors) vector += num
    this
  }

  def +=(index: Int, num: Double): LearningParameters = {
    require(index <= vectors.size - 1)
    vectors(index) += num
    this
  }

  override def +(params: LearningParameters): LearningParameters = {
    params match {
      case VectorList(vs: ListBuffer[EuclideanVector]) =>
        VectorList(
          for ((vector: EuclideanVector, v: EuclideanVector) <- vectors zip vs)
            yield (vector + v).asInstanceOf[EuclideanVector]
        )
      case v: Vector =>
        VectorList(for (vector: EuclideanVector <- vectors) yield (vector + v).asInstanceOf[EuclideanVector])
      case _ => throw new RuntimeException("The provided LearningParameter Object is non-compatible " +
        "for addition with a VectorList Object.")
    }
  }

  def +(index: Int, params: EuclideanVector): LearningParameters = {
    require(index <= vectors.size - 1)
    VectorList(
      for ((vector: EuclideanVector, i: Int) <- vectors.zipWithIndex)
        yield (if (i != index) vector else vector + params).asInstanceOf[EuclideanVector]
    )
  }

  override def +=(params: LearningParameters): LearningParameters = {
    params match {
      case VectorList(vs: ListBuffer[EuclideanVector]) =>
        for ((vector: EuclideanVector, v: EuclideanVector) <- vectors zip vs) vector += v
        this
      case v: Vector =>
        for (vector: EuclideanVector <- vectors) vector += v
        this
      case _ => throw new RuntimeException("The provided LearningParameter Object is non-compatible " +
        "for addition with a VectorList Object.")
    }
  }

  def +=(index: Int, params: EuclideanVector): LearningParameters = {
    require(index <= vectors.size - 1)
    vectors(index) += params
    this
  }
  override def -(num: Double): LearningParameters = this + (-num)

  def -(index: Int, num: Double): LearningParameters = this + (index, -num)

  override def -=(num: Double): LearningParameters = this += (-num)

  def -=(index: Int, num: Double): LearningParameters = this += (index, -num)

  override def -(params: LearningParameters): LearningParameters = {
    params match {
      case VectorList(vs: ListBuffer[EuclideanVector]) =>
        this + VectorList(for (v <- vs) yield (v * -1.0).asInstanceOf[EuclideanVector])
      case v: EuclideanVector =>
        this + (v * -1.0)
      case _ => throw new RuntimeException("The provided LearningParameter Object is non-compatible " +
        "for subtraction with a VectorList Object.")
    }
  }

  def -(index: Int, params: EuclideanVector): LearningParameters =
    this + (index, (params * -1.0).asInstanceOf[EuclideanVector])

  override def -=(params: LearningParameters): LearningParameters = {
    params match {
      case VectorList(vs: ListBuffer[EuclideanVector]) =>
        this += VectorList(for (v <- vs) yield (v * -1.0).asInstanceOf[EuclideanVector])
      case v: EuclideanVector =>
        this += (v * -1.0)
      case _ => throw new RuntimeException("The provided LearningParameter Object is non-compatible " +
        "for subtraction with a VectorList Object.")
    }
  }

  def -=(index: Int, params: EuclideanVector): LearningParameters =
    this += (index, (params * -1.0).asInstanceOf[EuclideanVector])

  override def *(num: Double): LearningParameters =
    VectorList(for (vector: EuclideanVector <- vectors) yield (vector * num).asInstanceOf[EuclideanVector])

  def *(index: Int, num: Double): LearningParameters = {
    require(index <= vectors.size - 1)
    VectorList(for ((vector: EuclideanVector, i: Int) <- vectors.zipWithIndex)
      yield (if (i != index) vector else vector * num).asInstanceOf[EuclideanVector]
    )
  }

  override def *=(num: Double): LearningParameters = {
    for (vector: EuclideanVector <- vectors) vector *= num
    this
  }

  def *=(index: Int, num: Double): LearningParameters = {
    require(index <= vectors.size - 1)
    vectors(index) *= num
    this
  }

  override def /(num: Double): LearningParameters = this * (1.0 / num)

  def /(index: Int, num: Double): LearningParameters = this * (index, 1.0 / num)

  override def /=(num: Double): LearningParameters = this *= (1.0 / num)

  def /=(index: Int, num: Double): LearningParameters = this *= (index, 1.0 / num)

  override def getCopy: LearningParameters = {
    val newVectors: ListBuffer[EuclideanVector] = ListBuffer[EuclideanVector]()
    for (vector <- vectors)
      newVectors += vector.getCopy.asInstanceOf[EuclideanVector]
    VectorList(newVectors)
  }

  override def flatten: BreezeDenseVector[Double] =
    (for (vector: EuclideanVector <- vectors) yield vector.flatten).reduce((x, y) => BreezeDenseVector.vertcat(x, y))

  override def extractParams: (LearningParameters, Boolean) => SerializableParameters = {
    (params: LearningParameters, sparse: Boolean) =>
      try {
        assert(params.isInstanceOf[VectorList])
        val bucket: Bucket = Bucket(0, params.asInstanceOf[VectorList].size - 1)
        wrappedParams(
          (for (vector: EuclideanVector <- vectors) yield vector.getSizes).flatten.toArray,
          null,
          params.asInstanceOf[VectorList].slice(bucket, sparse)
        )
      } catch {
        case _: Throwable =>
          throw new RuntimeException("Something happened while extracting the VectorList learning parameters.")
      }
  }

  override def extractDivParams: (LearningParameters, Array[_]) => Array[Array[SerializableParameters]] = {
    (params: LearningParameters, args: Array[_]) =>
      try {
        assert(params.isInstanceOf[VectorList] && args.length == 2)
        val (sparse, quantiles) = extractSparseQuantiles(args)
        val wrapped = ListBuffer[Array[SerializableParameters]]()
        for (buckets: Array[Bucket] <- quantiles)
          wrapped append {
            val hubWrap = for (bucket: Bucket <- buckets)
              yield wrappedParams(null, bucket, params.asInstanceOf[VectorList].slice(bucket, sparse))
            hubWrap.head.setSizes((for (vector: EuclideanVector <- vectors) yield vector.getSizes).flatten.toArray)
            hubWrap.asInstanceOf[Array[SerializableParameters]]
          }
        wrapped.toArray
      } catch {
        case _: Throwable =>
          throw new RuntimeException("Something happened while extracting the divided VectorList learning parameters.")
      }
  }

  override def generateParameters(pDesc: ParameterDescriptor): LearningParameters = {
    try {
      new VectorList(unwrapData(pDesc.getParamSizes, toDense(pDesc.getParams).data))
    } catch {
      case _: Throwable =>
        throw new RuntimeException("Something happened while deserializing the VectorList learning parameters.")
    }
  }

}
