package mlAPI.mlworkers.worker

import mlAPI.math.{DenseVector, SparseVector}
import mlAPI.parameters.utils.{Bucket, ParameterDescriptor, WrappedVectoredParameters}
import mlAPI.parameters.{LearningParameters, VectoredParameters, utils}
import mlAPI.protocols.{DoubleWrapper, LongWrapper, QuantilesWrapper}

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

/**
 * An abstract base class of an Online Machine Learning worker with a vectored model. The Machine Learning model
 * parameters vector is divided into smaller vectors in two successive ways. First it is divided equally to as many
 * smaller vectors as the hubs that are connected to the distributed learning topology. Then each of the smaller vectors
 * associated to each hub are divided into even smaller vectors so that all messages would not contain more than the
 * maximum number of parameters that each message is constrained to contain. The updateModel method implemented in this
 * class is responsible for collecting all the smaller sub vectors from each hub and reconstructing the whole learning
 * parameters vector in order to update the local model.
 *
 * @tparam ProxyIfc The remote interface of the Parameter Server.
 * @tparam QueryIfc The remote interface of the querier.
 */
abstract class VectoredWorker[ProxyIfc, QueryIfc]() extends MLWorker[ProxyIfc, QueryIfc] {

  /** A helper variable to detect how many hub sub vector models have been reconstructed. */
  private var readySplitModels: Int = 0

  /** The number of sub vector models that each hub exchanges with the current worker. */
  protected var splits = new mutable.HashMap[Int, Int]()

  /** The range boundaries used to split the vectored model into pieces to send to each hub. */
  protected var quantiles: Array[Array[Bucket]] = _

  /** A TreeMap with the parameter splits. If a model is too large to be transmitted within the Flink network, then
   * it is splitted into multiple smaller messages. This data structure is used to reconstruct the model from all the
   * received messages. */
  protected var parameterTree = new mutable.TreeMap[Int, mutable.TreeMap[(Int, Int), ParameterDescriptor]]()

  // =============================================== Getters ===========================================================

  def getQuantiles: Array[Array[Bucket]] = quantiles

  def getParameterTree: mutable.TreeMap[Int, mutable.TreeMap[(Int, Int), ParameterDescriptor]] = parameterTree

  // =============================================== Setters ===========================================================

  def setQuantiles(quantiles: Array[Array[Bucket]]): Unit = this.quantiles = quantiles

  def setParameterTree(parameterTree: mutable.TreeMap[Int, mutable.TreeMap[(Int, Int), ParameterDescriptor]]): Unit =
    this.parameterTree = parameterTree

  // ================================= ML vectored worker basic operations =============================================

  /** Calculates the drift of the local model compared to
   * the last global model received by the ML worker.
   *
   * @return The delta parameters.
   */
  def getDeltaVector: LearningParameters =
    getMLPipelineParams.get.asInstanceOf[VectoredParameters] - getGlobalParams.get.asInstanceOf[VectoredParameters]

  /** Divides a bucket range into smaller bucket ranges of maximum size [[maxMsgParams]]. */
  def divideBucket(b: Bucket): Array[Bucket] = {
    val slices = (b.getLength / maxMsgParams + {
      if (b.getLength % maxMsgParams == 0) 0 else 1
    }).toInt
    val buckets = ListBuffer[Bucket]()
    for (i: Int <- 0 until slices) {
      val start = i * maxMsgParams + b.getStart
      val end = {
        (start - 1) + {
          if (i == slices - 1)
            b.getLength % maxMsgParams
          else
            maxMsgParams
        }
      }
      buckets += Bucket(start, end)
    }
    buckets.toArray
  }

  /** Creates the bucket ranges used for splitting the local machine learning model into smaller vectors. */
  def generateQuantiles(): Unit = {
    require(mlPipeline.getLearner.getParameters.isDefined)

    val numberOfBuckets: Int = getNumberOfHubs
    val bucketSize: Int = mlPipeline.getLearner.getParameters.get.getSize / numberOfBuckets
    val remainder: Int = mlPipeline.getLearner.getParameters.get.getSize % numberOfBuckets

    @scala.annotation.tailrec
    def createRanges(index: Int, remainder: Int, quantiles: ListBuffer[Array[Bucket]]): Array[Array[Bucket]] = {
      if (index == numberOfBuckets)
        return quantiles.toArray
      quantiles.append(
        if (index == 0)
          divideBucket(Bucket(0, if (remainder > 0) bucketSize else bucketSize - 1))
        else
          divideBucket(
            Bucket(
              quantiles(index - 1).last.getEnd + 1,
              quantiles(index - 1).last.getEnd + {
                if (remainder > 0) bucketSize + 1 else bucketSize
              }
            )
          )
      )
      createRanges(index + 1, if (remainder > 0) remainder - 1 else remainder, quantiles)
    }

    quantiles = createRanges(0, remainder, ListBuffer[Array[Bucket]]())
    if (getNodeId == 0)
      for ((humSplits, hubId) <- quantiles.zipWithIndex)
        splits.put(hubId, humSplits.length)

    for (quantile <- quantiles)
      println(quantile.mkString("Array(", ", ", ")"))
    println(splits)
  }

  /** Converts the model into a Serializable POJO case class to be send over the Network. */
  override def ModelMarshalling(sparse: Boolean = false, sendSizes: Boolean = false, model: LearningParameters)
  : Array[Array[ParameterDescriptor]] = {
    try {
      val wrappedModel = model.extractDivParams(model, Array(sparse, quantiles))
      val descriptors = ListBuffer[Array[ParameterDescriptor]]()
      for (hubWrappedModel <- wrappedModel) {
        descriptors append {
          val hubDescriptors = ListBuffer[ParameterDescriptor]()
          for (wrappedModel <- hubWrappedModel)
            hubDescriptors append
              utils.ParameterDescriptor(
                if (wrappedModel.asInstanceOf[WrappedVectoredParameters].getSizes != null)
                  wrappedModel.asInstanceOf[WrappedVectoredParameters].getSizes
                else
                  null,
                wrappedModel.asInstanceOf[WrappedVectoredParameters].getData,
                wrappedModel.asInstanceOf[WrappedVectoredParameters].getBucket,
                null,
                null,
                null
              )
          if (sendSizes) {
            hubDescriptors.head.setMiscellaneous(Array(QuantilesWrapper(quantiles)))
            hubDescriptors.head.setFitted(LongWrapper(processedData))
          } else {
            hubDescriptors.last.setFitted(LongWrapper(processedData))
            hubDescriptors.head.paramSizes = null
          }
          hubDescriptors.toArray
        }
      }
      descriptors.toArray
    } catch {
      case _: Throwable =>
        generateQuantiles()
        ModelMarshalling(sparse, sendSizes, model)
    }
  }

  /** A private method for reconstructing the new local model from all the reconstructed sub vectors send by the hubs. */
  private def reconstructModel: ParameterDescriptor = {
    val model: ParameterDescriptor = parameterTree.toArray.sortBy(_._1).map(x => x._2.head._2)
      .fold[ParameterDescriptor](new ParameterDescriptor())(
        (acc, splitModel) => {
          if (acc.paramSizes == null)
            acc.setParamSizes(splitModel.getParamSizes)
          else
            assert(acc.getParamSizes.zip(splitModel.getParamSizes).map(x => x._1 == x._2).forall(y => y))
          acc.setParams(
            if (acc.getParams == null)
              DenseVector(splitModel.getParams.asInstanceOf[DenseVector].data)
            else
              DenseVector(
                acc.getParams.asInstanceOf[DenseVector].data ++ splitModel.getParams.asInstanceOf[DenseVector].data
              )
          )
          acc
        }
      )
    for (hub <- parameterTree.keys)
      parameterTree(hub).clear()
    readySplitModels = 0
    model
  }

  /** Private method for reconstructing a sub vector model send by a hub. */
  private def reconstructHubSubVector(): Unit = {
    val modelSlice: ParameterDescriptor = parameterTree(getCurrentCaller).values
      .fold(new ParameterDescriptor())(
        (acc, desc) => {
          if (acc.paramSizes == null)
            acc.setParamSizes(desc.getParamSizes)
          acc.setParams(
            if (acc.getParams == null)
              desc.getParams match {
                case dense: DenseVector => dense
                case sparse: SparseVector => sparse.toDenseVector
              }
            else
              DenseVector(
                acc.getParams.asInstanceOf[DenseVector].data ++
                  (desc.getParams match {
                    case dense: DenseVector => dense
                    case sparse: SparseVector => sparse.toDenseVector
                  }).data
              )
          )
          acc
        }
      )
    parameterTree(getCurrentCaller).clear()
    parameterTree(getCurrentCaller).put((0, 0), modelSlice)
    readySplitModels += 1
  }

  /** A private method that updates the parameter tree data structure with some sub vector received by a hub. */
  def updateParameterTree(slices: Int, mDesc: ParameterDescriptor, update: ParameterDescriptor => Unit): Unit = {
    if (!parameterTree.contains(getCurrentCaller))
      parameterTree.put(getCurrentCaller, new mutable.TreeMap[(Int, Int), ParameterDescriptor]())
    parameterTree(getCurrentCaller).put((mDesc.getBucket.getStart.toInt, mDesc.getBucket.getEnd.toInt), mDesc)
    if (mDesc.getFitted != null)
      mlPipeline.setFittedData(mDesc.getFitted.getLong)
    if (slices == parameterTree(getCurrentCaller).size) {
      reconstructHubSubVector()
      if (readySplitModels == getNumberOfHubs)
        if (isWarmedUp)
          update(reconstructModel)
        else
          warmModel(reconstructModel)
    }
  }

}
