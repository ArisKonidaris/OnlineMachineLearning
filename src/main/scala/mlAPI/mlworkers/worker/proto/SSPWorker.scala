package mlAPI.mlworkers.worker.proto

import BipartiteTopologyAPI.annotations.{InitOp, ProcessOp, QueryOp}
import ControlAPI.{QueryResponse, Request}
import mlAPI.math.Point
import mlAPI.mlworkers.interfaces.Querier
import mlAPI.mlworkers.worker.{MLWorker, VectoredWorker}
import mlAPI.parameters.VectoredParameters
import mlAPI.parameters.utils.{ParameterDescriptor, WrappedVectoredParameters}
import mlAPI.protocols.periodic.{PushPull, RemoteLearner}
import mlAPI.protocols.{IntWrapper, LongWrapper}
import mlAPI.utils.Parsing

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.collection.JavaConverters._

case class SSPWorker(override protected var maxMsgParams: Int = 10000)
  extends VectoredWorker[PushPull, Querier] with RemoteLearner {

  println("Stale Synchronous Worker initialized.")

  protocol = "Stale Synchronous Parallel"

  /** The local clock. */
  var clock: Long = 0

  /** The staleness parameter. */
  var s: Int = 3

  /** The min variable if the SSP protocol. */
  var min: Long = 0

  /** Initialization method of the Machine Learning worker node. */
  @InitOp
  def init(): Unit = {
    if (getNodeId != 0)
      pull()
  }

  /** Requesting the global model from the parameter server(s). */
  def pull(): Unit = {
    setWarmed(false)
    blockStream()
    for (i <- 0 until getNumberOfHubs)
      getProxy(i).pull()
  }

  /** The consumption of a data point by the Machine Learning worker.
   *
   * @param data A data point to be fitted to the model.
   */
  @ProcessOp
  def receiveTuple(data: Point): Unit = {
    fit(data)
    if (processedData >= getMiniBatchSize * miniBatches)
      if (!isWarmedUp && getNodeId == 0) {
        val warmupModel = {
          val wrapped = getMLPipelineParams.get.extractParams(
            getMLPipelineParams.get.asInstanceOf[VectoredParameters],
            false
          ).asInstanceOf[WrappedVectoredParameters]
          ParameterDescriptor(wrapped.getSizes, wrapped.getData, null, null, null, null)
        }
        setWarmed(true)
        setGlobalModelParams(warmupModel)
        for ((hubSubVector, index: Int) <- ModelMarshalling(sendSizes = true, model = getMLPipelineParams.get).zipWithIndex)
          for (slice <- hubSubVector)
            getProxy(index).push(slice)
        processedData = 0
      } else {
        incrementClock()
        if (clock - min > s)
          pushPull()
        else
          for ((hubSubVector, index: Int) <- ModelMarshalling(model = getDeltaVector).zipWithIndex) {
            hubSubVector.last.setMiscellaneous(Array(LongWrapper(clock)))
            for (slice <- hubSubVector)
              getProxy(index).push(slice)
          }
        processedData = 0
      }
  }

  /** Pushing the local model to the parameter server(s) and waiting for the new global model. */
  def pushPull(): Unit = {
    for ((hubSubVector: Array[ParameterDescriptor], index: Int) <- ModelMarshalling(model = getDeltaVector).zipWithIndex) {
      hubSubVector.last.setMiscellaneous(Array(LongWrapper(clock)))
      for (slice <- hubSubVector)
        getProxy(index).pushPull(slice).toSync(updateModel)
    }
    processedData = 0
  }

  /**
   * A method called each type the new global model
   * (or a slice of it) arrives from a parameter server.
   */
  override def updateModel(mDesc: ParameterDescriptor): Unit = {
    try {
      val spt: Int = {
        try {
          splits(getCurrentCaller)
        } catch {
          case _: Throwable =>
            assert(mDesc.getMiscellaneous != null, mDesc.getMiscellaneous.head.isInstanceOf[IntWrapper])
            splits.put(getCurrentCaller, mDesc.getMiscellaneous.head.asInstanceOf[IntWrapper].getInt)
            splits(getCurrentCaller)
        }
      }
      if (getNumberOfHubs == 1)
        if (spt == 1)
          if (isWarmedUp)
            updateModels(mDesc)
          else
            warmModel(mDesc)
        else
          updateParameterTree(spt, mDesc, updateModels)
      else
        updateParameterTree(spt, mDesc, updateModels)

      if (mDesc.getMiscellaneous != null && mDesc.getMiscellaneous(0).isInstanceOf[LongWrapper])
        min = mDesc.getMiscellaneous(0).asInstanceOf[LongWrapper].getLong

    } catch {
      case e: Throwable =>
        e.printStackTrace()
        throw new RuntimeException("Something went wrong while updating the local model of worker " +
          getNodeId + " of MLPipeline " + getNetworkID + ".")
    }
  }

  /** This method responds to a query for the Machine Learning worker.
   *
   * @param predicates The predicated of the query.
   */
  @QueryOp
  def query(queryId: Long, queryTarget: Int, predicates: (Double, Array[Point])): Unit = {
    val pj = mlPipeline.generatePOJO
    val score = getGlobalPerformance(ListBuffer(predicates._2: _ *))
    if (queryId == -1)
      getQuerier.sendQueryResponse(
        new QueryResponse(-1,
          queryTarget,
          null,
          null,
          null,
          processedData,
          null,
          predicates._1,
          score)
      )
    else
      getQuerier.sendQueryResponse(
        new QueryResponse(queryId, queryTarget, pj._1.asJava, pj._2, protocol, pj._3, pj._4, pj._5, score)
      )
  }

  def getS: Int = s

  def setS(s: Int): Unit = this.s = s

  def getClock: Long = clock

  def setClock(clock: Long): Unit = this.clock = clock

  def incrementClock(inc: Long = 1): Unit = {
    if (clock + inc <= Int.MaxValue)
      clock += inc
    else
      clock = 0
  }

  override def configureWorker(request: Request): MLWorker[PushPull, Querier] = {
    super.configureWorker(request)

    // Setting the ML node parameters.
    val config: mutable.Map[String, AnyRef] = request.getTrainingConfiguration.asScala

    // Set the s parameter.
    if (config.contains("s")) {
      setS(
        try {
          Parsing.IntegerParsing(config, "s", 3)
        } catch {
          case _: Throwable => 3
        }
      )
    }

    this
  }

}
