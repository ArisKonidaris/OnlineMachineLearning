package mlAPI.mlworkers.worker.proto

import BipartiteTopologyAPI.annotations.{InitOp, ProcessOp, QueryOp}
import ControlAPI.{Prediction, QueryResponse, Request}
import mlAPI.math.{ForecastingPoint, LabeledPoint, LearningPoint, TrainingPoint, UnlabeledPoint, UsablePoint}
import mlAPI.mlworkers.interfaces.Querier
import mlAPI.mlworkers.worker.{MLWorker, VectoredWorker}
import mlAPI.parameters.VectoredParameters
import mlAPI.parameters.utils.{ParameterDescriptor, WrappedVectoredParameters}
import mlAPI.protocols.{DoubleWrapper, IntWrapper}
import mlAPI.protocols.dynamic._
import mlAPI.safezones.{SafeZone, VarianceSafeZone}
import mlAPI.utils.Parsing

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.collection.JavaConverters._

case class FGMWorker(private var safeZone: SafeZone = VarianceSafeZone(),
                     override protected var maxMsgParams: Int = 10000)
  extends VectoredWorker[FGMHubInterface, Querier] with FGMRemoteLearner {

  protocol = "FGM-protocol"

  /** The quantum of the FMG distributed learning protocol. */
  private var quantum: Double = _

  /** The counter of the FGM worker. */
  private var counter: Long = 0

  /** The value of the safe zone function. */
  private var zeta: Double = 0

  /** A temp variable for the value of the safe zone function. */
  private var tempZeta: Double = 0

  /** A flag variable determining the status of the current subround. */
  private var activeSubRound: Boolean = true

  /** The active subround. */
  private var subRound: Long = 0

  /** The current FGM round. */
  private var round: Long = 0

  /**
   * A flag determining if the coordinator requested the local model when it is not yet initialized or received by this
   * worker in the first place.
   */
  var pendingModel: Boolean = false

  /** A flag determining whether the initial phi of a new round has been sent to the hub. */
  var sentPhi: Boolean = false

  /** Initialization method of the Machine Learning worker node. */
  @InitOp
  def init(): Unit = {
    println("Network: " + getNetworkID + "| FGM Worker " + getNodeId + " initialized.")
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

  /** A method for training the FGM worker on a training data point. */
  def train(data: LearningPoint): Unit = {
    fit(data)
    if (!isWarmedUp) {
      if (getNodeId == 0 && (processedData % (getMiniBatchSize * miniBatches) == 0)) {
        val warmupModel = {
          val wrapped = getMLPipelineParams.get.extractParams(
            getMLPipelineParams.get.asInstanceOf[VectoredParameters],
            false
          ).asInstanceOf[WrappedVectoredParameters]
          ParameterDescriptor(wrapped.getSizes, wrapped.getData, null, null, null, null)
        }
        setWarmed(true)
        setGlobalModelParams(warmupModel)
        val sWarmupModel = ModelMarshalling(sendSizes = true, model = getMLPipelineParams.get)
        for ((hubSubVector: Array[ParameterDescriptor], index: Int) <- sWarmupModel.zipWithIndex)
          for (slice <- hubSubVector)
            getProxy(index).endWarmup(slice)
        processedData = 0
        blockStream()
      }
    } else {
      if (activeSubRound && subRound > 0 && (processedData % (getMiniBatchSize * miniBatches) == 0)) {
        val distFromBoundary: Long = scala.math.floor((zeta - getZeta) / quantum).toLong
        val increment: Long = distFromBoundary - counter
        if (increment > 0) {
          counter = distFromBoundary
          getProxy(0).receiveIncrement(Increment(increment, subRound))
        }
      }
    }
  }

  /** The consumption of a data point by the Machine Learning FGM worker.
   *
   * @param data A data point to be fitted to the model.
   */
  @ProcessOp
  def receiveTuple(data: UsablePoint): Unit = {
    assert(isWarmedUp || (!isWarmedUp && getNodeId == 0))
    data match {
      case TrainingPoint(trainingPoint) => train(trainingPoint)
      case ForecastingPoint(forecastingPoint) =>
        val prediction: Double = {
          try {
            globalModel.predict(forecastingPoint) match {
              case Some(prediction: Double) => prediction
              case None => Double.NaN
            }
          } catch {
            case _: Throwable => Double.NaN
          }
        }
        getQuerier.sendQueryResponse(new Prediction(getNetworkID(), forecastingPoint.toDataInstance, prediction))
    }
  }

  /** Resets the local variables of the worker for starting a new FGM round. */
  def reset(): Unit = {
    zeta = safeZone.newRoundZeta() // Reset the safe zone function to E.
    counter = 0 // Reset the worker counter.
    this.quantum = zeta / 2.0 // Update the quantum.
    activeSubRound = true // Reset the active sub round flag.
    subRound += 1 // Update the current running subround.
    round += 1 // Update the current running round.
    if (getNodeId == 0)
      sentPhi = false // Reset the sent phi flag.
    println("Network: " + getNetworkID + "| Worker: " + getNodeId + " started new round: " + round + ", zeta: " + zeta)
  }

  /** Sending the local model to the coordinator. */
  override def sendLocalDrift(): Unit = {
    assert(!activeSubRound, getNodeId + " failed at round " + round)
    for ((hubSubVec: Array[ParameterDescriptor], index: Int) <- ModelMarshalling(model = getDeltaVector).zipWithIndex)
      for (slice <- hubSubVec)
        getProxy(index).receiveLocalDrift(slice).toSync(newRound)
    processedData = 0
  }

  /** Sending the safe zone function value to the coordinator. */
  override def requestZeta(): Unit = {
    getMLPipelineParams match {
      case None => pendingModel = true
      case Some(_) =>
//        println("Zeta has been requested from worker " + getNodeId)
        activeSubRound = false // Stop calculating and sending increments.
        getProxy(0).receiveZeta(
          if (getNodeId == 0 && !sentPhi) {
            sentPhi = true
            ZetaValue(getZeta, DoubleWrapper(getNumberOfSpokes * zeta))
          } else
            new ZetaValue(getZeta)
        )
    }
  }

  /** Receive the new quantum from the coordinator in order to resume the FGM round.
   *
   * @param quantum The new quantum sent by the coordinator to start a new subround.
   * */
  override def receiveQuantum(quantum: Quantum): Unit = {
    counter = 0 // Reset the worker counter.
    this.quantum = quantum.getValue // Update the quantum.
    zeta = getZeta // Update zeta.
    activeSubRound = true // Resume calculating and sending increments.
    subRound += 1 // Update the current running subround.
  }

  /**
   * A method called each time the new global model (or a slice of it) arrives from a parameter server.
   *
   * @param mDesc The piece of the new model sent by the hub.
   */
  override def newRound(mDesc: ParameterDescriptor): Unit = {
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
      if (mDesc.getParams != null)
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
      else
        assertWarmup()

    } catch {
      case e: Throwable =>
        e.printStackTrace()
        throw new RuntimeException("Something went wrong while resetting worker " +
          getNodeId + " of MLPipeline " + getNetworkID + " for a new FGM round.")
    }
  }

  override def warmModel(mDesc: ParameterDescriptor): Unit = {
    super.warmModel(mDesc)
    reset()
    if (pendingModel) {
      pendingModel = false
      requestZeta()
    }
  }

  override def updateModels(mDesc: ParameterDescriptor): Unit = {
    super.updateModels(mDesc)
    reset()
    if (pendingModel) {
      pendingModel = false
      requestZeta()
    }
  }

  def assertWarmup(): Unit = {
    assert(getNodeId == 0 && getMLPipeline.getFittedData == getMiniBatchSize * miniBatches && isBlocked)
    reset()
    unblockStream()
  }

  def setSafeZone(safeZone: SafeZone): Unit = this.safeZone = safeZone

  def getSafeZone: SafeZone = {
    val value = safeZone
    value
  }

  def getZeta: Double = {
    safeZone.zeta(
      getGlobalParams.asInstanceOf[Option[VectoredParameters]].get,
      getMLPipelineParams.asInstanceOf[Option[VectoredParameters]].get
    )
  }

  override def configureWorker(request: Request): MLWorker[FGMHubInterface, Querier] = {
    super.configureWorker(request)

    // Setting the ML node parameters.
    val config: mutable.Map[String, AnyRef] = request.getTrainingConfiguration.asScala

    if (config.contains("safeZone")) {
      try {
        setSafeZone(
          config("safeZone").asInstanceOf[String] match {
            case "ModelVariance" =>
              if (config.contains("threshold"))
                VarianceSafeZone(Parsing.DoubleParsing(config, "threshold", 0.008))
              else
                VarianceSafeZone()
            case _ => VarianceSafeZone()
          }
        )
      } catch {
        case _: Throwable => VarianceSafeZone()
      }
    }

    this
  }

  /** This method responds to a query for the Machine Learning worker.
   *
   * @param predicates The predicated of the query.
   */
  @QueryOp
  def query(queryId: Long, queryTarget: Int, predicates: (Double, Array[UsablePoint])): Unit = {
    val pj = mlPipeline.generatePOJO
    val testSet: Array[LearningPoint] = predicates._2.map {
      case TrainingPoint(trainingPoint) => trainingPoint
      case ForecastingPoint(forecastingPoint) => forecastingPoint
      case labeledPoint: LabeledPoint => labeledPoint
      case unlabeledPoint: UnlabeledPoint => unlabeledPoint
    }
    val score = getGlobalPerformance(ListBuffer(testSet: _ *))
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