package mlAPI.mlParameterServers

import BipartiteTopologyAPI.futures.Response
import breeze.linalg.{DenseVector => BreezeDenseVector}
import mlAPI.math.{DenseVector, SparseVector}
import mlAPI.parameters.utils.{Bucket, ParameterDescriptor}
import mlAPI.preprocessing.RunningMean
import mlAPI.protocols.{DoubleWrapper, IntWrapper, LongWrapper, QuantilesWrapper}

import scala.annotation.tailrec
import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.collection.JavaConverters._

/**
 * An abstract base class of a Machine Learning Parameter Server that keeps the global model in a flat vector.
 *
 * @tparam WorkerIfc The remote interface of the Machine Learning worker.
 * @tparam QueryIfc  The remote interface of the querier.
 */
abstract class VectoredPS[WorkerIfc, QueryIfc] extends MLParameterServer[WorkerIfc, QueryIfc] {

  /** A flag indicating if the network is warmed up. */
  private var warmed: Boolean = false

  /** A helping counter. */
  var warmupCounter: Int = 0

  /** A variable indicating the number of workers of the distributed learning procedure. */
  var parallelism: Int = _

  /** Bla bla. */
  var jurisdiction: (Int, Array[Bucket]) = _

  /** The range boundaries used to split the vectored model into pieces. */
  var quantiles: mutable.Map[Int, Array[Bucket]] = _

  /** The size of the total global model. */
  var paramSizes: Array[Int] = _

  /** A data structure for reconstructing the drift vectors send by the workers. */
  var parameterTree = new mutable.HashMap[Int, mutable.TreeMap[(Int, Int), ParameterDescriptor]]()

  /** A variable for holding a reconstructed vector sent by a worker. */
  var reconstructedVectorSlice: BreezeDenseVector[Double] = _

  /** The actual global model of the distributed learning procedure. */
  var globalVectorSlice: BreezeDenseVector[Double] = _

  def updateParameterTree(pDesc: ParameterDescriptor): Boolean = {
    try {
      assert(quantiles != null && paramSizes != null)
    } catch {
      case _: Throwable => confirmWarmup(pDesc)
    }
    if (!parameterTree.contains(getCurrentCaller))
      parameterTree.put(getCurrentCaller, new mutable.TreeMap[(Int, Int), ParameterDescriptor]())
    if (pDesc.getFitted != null)
      incrementNumberOfFittedData(pDesc.getFitted.getLong)
    if (pDesc.getMiscellaneous != null)
      if (pDesc.getMiscellaneous.length == 1 && pDesc.getMiscellaneous.head.isInstanceOf[DoubleWrapper])
        getRoundLoss.update(pDesc.getMiscellaneous.head.asInstanceOf[DoubleWrapper].getDouble)
      else
        if (pDesc.getMiscellaneous.last.isInstanceOf[DoubleWrapper])
          getRoundLoss.update(pDesc.getMiscellaneous.last.asInstanceOf[DoubleWrapper].getDouble)
    parameterTree(getCurrentCaller).put((pDesc.getBucket.getStart.toInt, pDesc.getBucket.getEnd.toInt), pDesc)
    if (jurisdiction._1 == parameterTree(getCurrentCaller).size) {
      reconstructedVectorSlice =
        BreezeDenseVector(
          parameterTree(getCurrentCaller).values
            .fold(new ParameterDescriptor())(
              (acc, desc) => {
                acc.setParams(
                  if (acc.params == null)
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
            ).getParams.asInstanceOf[DenseVector].data
        )
      parameterTree(getCurrentCaller).clear()
      true
    } else
      false
  }

  def confirmWarmup(mDesc: ParameterDescriptor): Unit = {
    assert(
      getCurrentCaller == 0 &&
        mDesc.getFitted.getLong > 0 &&
        globalVectorSlice == null &&
        mDesc.getParamSizes != null &&
        mDesc.getMiscellaneous.head.isInstanceOf[QuantilesWrapper]
    )
    paramSizes = mDesc.getParamSizes
    quantiles = collection.mutable.Map(
      mDesc.getMiscellaneous.head.asInstanceOf[QuantilesWrapper].getQuantiles
        .zipWithIndex.map(x => (x._2, x._1)).toMap
      .toSeq: _*
    )
    val q = quantiles(getNodeId).clone()
    jurisdiction = (q.length, normalizeBucket(q, ListBuffer[Bucket]()))
  }

  @tailrec
  private def normalizeBucket(buckets: Array[Bucket], result: ListBuffer[Bucket]): Array[Bucket] = {
    if (buckets.isEmpty)
      result.toArray
    else {
      if (result.isEmpty)
        result += Bucket(0, buckets.head.getEnd - buckets.head.getStart)
      else
        result += Bucket(result.last.getEnd + 1, result.last.getEnd + 1 + (buckets.head.getEnd - buckets.head.getStart))
      normalizeBucket(buckets.tail, result)
    }
  }

  /** This method sends the global model to the worker that requested it. */
  def sendModelToWorker(): Response[ParameterDescriptor]  = {
    assert (isWarmedUp)
    val model = serializableModel()
    protocolStatistics.updateModelsShipped()
    protocolStatistics.updateBytesShipped((for (slice <- model) yield slice.getSize).sum)
    fulfillPromises(model.toList.asJava)
  }

  def sendModelToWorkers(): Response[ParameterDescriptor]  = {
    assert (isWarmedUp)
    val model = serializableModel()
    protocolStatistics.updateModelsShipped(parallelism)
    protocolStatistics.updateBytesShipped(parallelism * (for (slice <- model) yield slice.getSize).sum)
    fulfillBroadcastPromises(model.toList.asJava)
  }

  /** Returns the serialized warmed up model that is to be broadcasted to all the workers. */
  def warmUpModel(): Array[ParameterDescriptor] = {
    warmed = true
    warmupCounter = 0
    val model = serializableModel()
    model.head.setParamSizes(paramSizes)
    model.head.setMiscellaneous(Array(IntWrapper(jurisdiction._1)))
    model
  }

  /** A marshalled model response divided into smaller vectors.
   *
   * @return The marshalled model response.
   * */
  def serializableModel(data: Array[Double] = globalVectorSlice.data): Array[ParameterDescriptor] = {
    val model = ListBuffer[ParameterDescriptor]()
    for ((normBucket, bucket) <- jurisdiction._2.zip(quantiles(getNodeId))) {
      model += ParameterDescriptor(
        null,
        DenseVector(data.slice(normBucket.getStart.toInt, normBucket.getEnd.toInt + 1)),
        bucket,
        null,
        null,
        null
      )
    }
    model.head.setFitted(LongWrapper(fitted))
    model.toArray
  }

  def isWarmedUp: Boolean = {
    val value = warmed
    value
  }

  def updateLearningCurve(): Unit = {
    learningCurve.append((roundLoss.getMean, fitted))
    roundLoss = RunningMean()
  }

}
