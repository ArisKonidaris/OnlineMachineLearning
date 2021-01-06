package mlAPI.mlworkers.worker

import BipartiteTopologyAPI.annotations.{InitOp, ProcessOp, QueryOp}
import ControlAPI.{QueryResponse, Request}
import mlAPI.math.Point
import mlAPI.mlworkers.interfaces.Querier
import mlAPI.parameters.{BreezeParameters, ParameterDescriptor, VectoredParameters}
import mlAPI.protocols.fgm.{FGMHubInterface, FGMRemoteLearner, Increment, Quantum, ZetaValue}
import mlAPI.safezones.{SafeZone, VarianceSafeZone}
import mlAPI.utils.Parsing

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ListBuffer

case class FGMWorker(private var safeZone: SafeZone = VarianceSafeZone())
  extends VectoredWorker[FGMHubInterface, Querier] with FGMRemoteLearner {

  protocol = "FGM-protocol"

  /** The quantum of the FMG distributed learning protocol. */
  var quantum: Double = _

  /** The counter of the FGM worker. */
  var counter: Long = 0

  /** The value of the safe zone function. */
  var zeta: Double = 0

  /** A temp variable for the value of the safe zone function. */
  var tempZeta: Double = 0

  /** A flag variable determining the status of the current subround. */
  var activeSubRound: Boolean = true

  /** The active subround. */
  var subRound: Long = 0

  /** Initialization method of the Machine Learning worker node. */
  @InitOp
  def init(): Unit = {
    println("Network: " + getNetworkID + "| FGM Worker " + getNodeId + " initialized.")
    if (getNodeId != 0) pull()
  }

  /** A method called each type the new global model arrives from the parameter server.
   *
   * @param modelDescriptor The new model and the new quantum send by the coordinator in order to start a new FGM round.
   * */
  override def reset(modelDescriptor: ParameterDescriptor): Unit = {
    updateLocalModel(modelDescriptor) // Update the model of the local learner.
    zeta = safeZone.newRoundZeta()
    counter = 0 // Reset the worker counter.
    quantum = modelDescriptor.miscellaneous.asInstanceOf[Quantum].getValue // Update the quantum.
    activeSubRound = true // Reset the active sub round flag.
    subRound += 1 // Update the current running subround.
    println("Network: " + getNetworkID + "| Worker: " + getNodeId + " started new round: " + subRound + ", zeta: " + zeta)
  }

  /** The consumption of a data point by the Machine Learning worker.
   *
   * @param data A data point to be fitted to the model.
   */
  @ProcessOp
  def receiveTuple(data: Point): Unit = {
    fit(data)
    if (activeSubRound && (processedData % (getMiniBatchSize * miniBatches) == 0)) {
      if (subRound > 0) { // Check distance from the safe zone boundary
        val z = safeZone.zeta(
          getGlobalParams.asInstanceOf[Option[VectoredParameters]].get,
          getMLPipelineParams.asInstanceOf[Option[VectoredParameters]].get
        )
//        println("Worker " + getNodeId + ", z: " + z)
        val distFromBoundary: Long = scala.math.floor((zeta - z) / quantum).toLong
//        println("Worker " + getNodeId + ", dist: " + distFromBoundary)
        val increment: Long = distFromBoundary - counter
        if (increment > 0) {
          counter = distFromBoundary
          getProxy(0).receiveIncrement(Increment(increment, subRound))
        }
//        println("Worker " + getNodeId + ", counter: " + counter)
      } else { // Warmup
        assert(getNodeId == 0)
        getProxy(0).endWarmup(
          {
            val model: Array[ParameterDescriptor] = ModelMarshalling(drift = false)
            assert(model.length == 1)
            model(0)
          }
        ).toSync(reset)
      }
    }
  }

  /** Sending the local model to the coordinator. */
  override def sendLocalDrift(): Unit = {
    assert(!activeSubRound)
    val serializedDrift: Array[ParameterDescriptor] = {
      if (processedData > 0)
        ModelMarshalling()
      else
        Array(ParameterDescriptor(null, null, null, null, null, 0))
    }
    processedData = 0
    getProxy(0).receiveLocalDrift(serializedDrift(0)).toSync(reset)
//    println("Worker " + getNodeId + " sent local drift.")
  }

  /** Sending the safe zone function value to the coordinator. */
  override def requestZeta(): Unit = {
    activeSubRound = false // Stop calculating and sending increments.
    tempZeta = safeZone.zeta(getGlobalParams.asInstanceOf[Option[VectoredParameters]].get,
      getMLPipelineParams.asInstanceOf[Option[VectoredParameters]].get
    )
    getProxy(0).receiveZeta(ZetaValue(tempZeta))
//    println("Worker " + getNodeId + " sent local zeta.")
  }

  /** Receive the new quantum from the coordinator in order to resume the FGM round.
   *
   * @param quantum The new quantum sent by the coordinator to start a new subround.
   * */
  override def receiveQuantum(quantum: Quantum): Unit = {
    counter = 0 // Reset the worker counter.
    this.quantum = quantum.getValue // Update the quantum.
    zeta = tempZeta // Update zeta.
    activeSubRound = true // Resume calculating and sending increments.
    subRound += 1 // Update the current running subround.
//    println("Worker " + getNodeId + " received new quantum.")
  }

  /** Pull the global model. */
  def pull(): Unit = getProxy(0).pullModel.toSync(reset)

  def setSafeZone(safeZone: SafeZone): Unit = this.safeZone = safeZone

  def getSafeZone: SafeZone = {
    val value = safeZone
    value
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
              if (config.contains("threshold")) {
                try {
                  VarianceSafeZone(Parsing.DoubleParsing(config, "threshold", 0.008))
                } catch {
                  case _: Throwable => VarianceSafeZone()
                }
              } else VarianceSafeZone()
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
