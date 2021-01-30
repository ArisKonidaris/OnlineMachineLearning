package mlAPI.mlworkers.worker.proto

import BipartiteTopologyAPI.annotations.{InitOp, ProcessOp, QueryOp}
import ControlAPI.{QueryResponse, Request}
import mlAPI.math.Point
import mlAPI.mlworkers.interfaces.Querier
import mlAPI.mlworkers.worker.{MLWorker, VectoredWorker}
import mlAPI.parameters.VectoredParameters
import mlAPI.parameters.utils.{ParameterDescriptor, WrappedVectoredParameters}
import mlAPI.protocols.dynamic.{GMHubInterface, GMRemoteLearner}
import mlAPI.protocols.IntWrapper
import mlAPI.utils.Parsing

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.collection.JavaConverters._

case class GMWorker(override protected var maxMsgParams: Int = 10000)
  extends VectoredWorker[GMHubInterface, Querier] with GMRemoteLearner {

  protocol = "GM-protocol"

  /** The radius of the admissible region. */
  var radius: Double = 0.0008

  /** A flag determining if the admissible region has been violated. */
  var violatedAR: Boolean = false

  /** A variable indicating the current round of the GM protocol. */
  var round: Long = 0

  /** Initialization method of the Machine Learning worker node. */
  @InitOp
  def init(): Unit = {
    assert(getNumberOfHubs == 1)
    println("Network: " + getNetworkID + "| GM Worker " + getNodeId + " initialized.")
    if (getNodeId != 0) {
      round += 1
      pull()
    }
  }

  /** Requesting the global model from the parameter server(s). */
  def pull(): Unit = {
    setWarmed(false)
    blockStream()
    for (i <- 0 until getNumberOfHubs)
      getProxy(i).pull()
  }

  /** The consumption of a data point by the Machine Learning FGM worker.
   *
   * @param data A data point to be fitted to the model.
   */
  @ProcessOp
  def receiveTuple(data: Point): Unit = {
    fit(data)
    if (!isWarmedUp) {
      if (processedData >= getMiniBatchSize * miniBatches) {
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
          for (slice <- ModelMarshalling(sendSizes = true, model = getMLPipelineParams.get)(0))
            getProxy(0).endWarmup(slice)
          processedData = 0
          blockStream()
        }
      }
    } else {
      if (!violatedAR && processedData % (getMiniBatchSize * miniBatches) == 0)
        if (math.pow(
          (
            getGlobalParams.asInstanceOf[Option[VectoredParameters]].get -
              getMLPipelineParams.asInstanceOf[Option[VectoredParameters]].get
            ).asInstanceOf[VectoredParameters].frobeniusNorm,
          2) > radius
        ) {
          violatedAR = true
          getProxy(0).violation()
        }
    }
  }

  /** Sending the local model to the coordinator. */
  override def sendLocalModel(): Unit = {
    println("Requested local model " + getNodeId)
    for (slice <- ModelMarshalling(model = getMLPipelineParams.get)(0))
      getProxy(0).receiveLocalModel(slice).toSync(updateModel)
  }

  /**
   * A method called each type the new global model
   * (or a slice of it) arrives from a parameter server.
   *
   * @param mDesc The piece of the new model and the new quantum send by the hub in order to start a new GM round.
   */
  override def updateModel(mDesc: ParameterDescriptor): Unit = {
    try {

      // Initialize splits.
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

      // Update models.
      if (spt == 1)
        if (isWarmedUp)
          if (mDesc.getParams != null)
            updateModels(mDesc)
          else
            assertWarmup()
        else
          warmModel(mDesc)
      else if (mDesc.getParams != null)
        updateParameterTree(spt, mDesc, updateModels)
      else
        assertWarmup()

    } catch {
      case e: Throwable =>
        e.printStackTrace()
        throw new RuntimeException("Something went wrong while resetting worker " +
          getNodeId + " of MLPipeline " + getNetworkID + " for a new GM round or subround.")
    }
  }

  override def updateModels(mDesc: ParameterDescriptor) {
    super.updateModels(mDesc)
    round += 1
    processedData = 0
    violatedAR = false
    println("Network: " + getNetworkID + "| Worker: " + getNodeId + " started new round: " + round)
  }

  def assertWarmup(): Unit = {
    assert(
      getNodeId == 0 &&
        getMLPipeline.getFittedData == getMiniBatchSize * miniBatches &&
        isBlocked
    )
    round += 1
    println("Network: " + getNetworkID + "| Worker: " + getNodeId + " started new round: " + round)
    unblockStream()
  }

  override def configureWorker(request: Request): MLWorker[GMHubInterface, Querier] = {
    super.configureWorker(request)

    // Setting the ML node parameters.
    val config: mutable.Map[String, AnyRef] = request.getTrainingConfiguration.asScala

    if (config.contains("radius"))
      radius = Parsing.DoubleParsing(config, "radius", 0.0008)

    this
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

}
