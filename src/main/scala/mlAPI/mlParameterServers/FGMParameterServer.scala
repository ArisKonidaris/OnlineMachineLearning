package mlAPI.mlParameterServers

import java.io.Serializable
import BipartiteTopologyAPI.annotations.{InitOp, MergeOp, ProcessOp, QueryOp}
import BipartiteTopologyAPI.futures.{BroadcastValueResponse, PromiseResponse, Response}
import ControlAPI.Request
import mlAPI.math.DenseVector
import mlAPI.mlworkers.interfaces.Querier
import mlAPI.parameters.ParameterDescriptor
import mlAPI.safezones.{SafeZone, VarianceSafeZone}
import breeze.linalg.{DenseVector => BreezeDenseVector}
import mlAPI.protocols.fgm.{FGMHubInterface, FGMRemoteLearner, FGMStatistics, Increment, Quantum, ZetaValue}
import mlAPI.utils.Parsing

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.collection.JavaConverters._

/**
 * A Parameter Server for FGM Distributed Learning with a global vector model.
 *
 * @param precision The precision of the FGM protocol.
 * @param safeZone  The safe zone function of the FGM protocol.
 */
case class FGMParameterServer(private var precision: Double = 0.01,
                              private var safeZone: SafeZone = VarianceSafeZone())
  extends VectoredPS[FGMRemoteLearner, Querier] with FGMHubInterface {

  println("FGM Hub initialized.")

  protocolStatistics = FGMStatistics()

  /** A synchronizer variable. */
  var sync: SyncProtocol = new SyncProtocol()

  /** The increment of a sub-round. */
  var inc: Long = 0

  /** Number of safe zones sent by the FGM coordinator. */
  var safeZonesSent: Long = 0

  /** The phi value of the FMG distributed learning protocol. */
  var phi: Double = _

  /** The quantum of the FMG distributed learning protocol. */
  var quantum: Double = _

  /** The smallest number the zeta function can reach. */
  var barrier: Double = _

  var drift: BreezeDenseVector[Double] = _

  /** Initialization method of the Parameter server node. */
  @InitOp
  def init(): Unit = sync.parallelism = getNumberOfSpokes

  /** This method prepares the FGM network for a new round. */
  def prepareNewRound(): Unit = {

    // Update the number of globally fitted data points.
    incrementNumberOfFittedData(sync.shippedDrifts.foldLeft(0L)((c, x) => c + x._2))

    // Resets
    sync.shippedDrifts.clear()
    drift *= 0.0
    sync.activeSubRound = true

    // Calculating the new phi, quantum and the minimum acceptable value for phi.
    phi = sync.parallelism * safeZone.newRoundZeta()
    quantum = phi / (2.0 * sync.parallelism)
    barrier = precision * phi

    // Increment the number of rounds, sub-rounds and number of shipped safe zones of the FGM distributed learning protocol.
    protocolStatistics.asInstanceOf[FGMStatistics].updateNumOfRounds()
    protocolStatistics.asInstanceOf[FGMStatistics].updateNumOfSubRounds()
    safeZonesSent += sync.parallelism
  }

  /** Starting a new round of the FGM distributed learning protocol.
   *
   * @return The new global model and quantum for the new round.
   * */
  def startRound(): BroadcastValueResponse[ParameterDescriptor] = {
    prepareNewRound()
    val packagedModel: ParameterDescriptor = sendModel().getValue
    protocolStatistics.updateModelsShipped(sync.parallelism)
    protocolStatistics.updateBytesShipped(sync.parallelism * packagedModel.getSize)
    fulfillBroadcastPromise(packagedModel)
  }

  /** Ending the warmup of the FGM distributed learning protocol.
   *
   * @param modelDescriptor The marshalled model sent by the first worker for warming up the FGM distributed
   *                        learning procedure.
   * @return The global model to be broadcasted to all the workers in order to start the FGM distributed
   *         learning procedure.
   * */
  override def endWarmup(modelDescriptor: ParameterDescriptor): Response[ParameterDescriptor] = {
    assertWarmup(modelDescriptor)
    protocolStatistics.updateModelsShipped()
    protocolStatistics.updateBytesShipped(modelDescriptor.getSize)
    protocolStatistics.updateNumOfBlocks()
    parametersDescription = modelDescriptor
    globalModel = deserializeVector(modelDescriptor)
    drift = 0.0 * globalModel
    makeBroadcastPromise(new PromiseResponse[ParameterDescriptor]())
    sync.shippedDrifts.put(0, modelDescriptor.getFitted)
    if (sync.shippedDrifts.size == sync.parallelism)
      startRound()
    else
      Response.noResponse()
  }

  /** A method invoked when a worker requests the global model.
   *
   * @return The global model.
   * */
  override def pullModel: Response[ParameterDescriptor] = {
    assert(getCurrentCaller != 0)
    protocolStatistics.updateNumOfBlocks()
    makeBroadcastPromise(new PromiseResponse[ParameterDescriptor]())
    sync.shippedDrifts.put(getCurrentCaller, 0)
    if (globalModel == null)
      Response.noResponse()
    else if (sync.shippedDrifts.size == sync.parallelism)
      startRound()
    else
      Response.noResponse()
  }

  /** Updating the global drift from the global model by adding the drift vector of the worker that called this method.
   *
   * @param remoteModelDescriptor The worker's drift from the global model.
   */
  def updateDrift(remoteModelDescriptor: ParameterDescriptor): Unit = {
    val remoteVector: BreezeDenseVector[Double] = deserializeVector(remoteModelDescriptor)
    drift += remoteVector
  }

  /** Receiving the drift of a worker.
   *
   * @param modelDescriptor The worker's drift from the global model.
   * @return The new global model.
   * */
  override def receiveLocalDrift(modelDescriptor: ParameterDescriptor): Response[ParameterDescriptor] = {
    protocolStatistics.updateModelsShipped()
    protocolStatistics.updateBytesShipped(modelDescriptor.getSize)
    protocolStatistics.updateNumOfBlocks()
    makeBroadcastPromise(new PromiseResponse[ParameterDescriptor]())
    sync.shippedDrifts.put(getCurrentCaller, modelDescriptor.getFitted)
    if (modelDescriptor.getFitted > 0)
      updateDrift(modelDescriptor)
    if (sync.shippedDrifts.size == sync.parallelism) {
      globalModel += (1.0 / (1.0 * sync.shippedDrifts.toArray.map(x => if (x._2 > 0) 1.0 else 0.0).sum)) * drift
      startRound()
    } else
      Response.noResponse()
  }

  /** Receiving an increment of a worker.
   *
   * @param increment The increment send by the worker.
   * */
  override def receiveIncrement(increment: Increment): Unit = {
    protocolStatistics.updateBytesShipped(increment.getSize)
    if (sync.activeSubRound && protocolStatistics.asInstanceOf[FGMStatistics].getNumOfSubRounds == increment.subRound) {
      inc += increment.increment
      if (inc > sync.parallelism) {
        sync.activeSubRound = false
        getBroadcastProxy.requestZeta()
      }
    }
//    println(inc)
  }

  /** Receiving the zeta safe zone function value of a worker.
   *
   * @param zeta The zeta safe zone function value sent by the worker.
   * */
  override def receiveZeta(zeta: ZetaValue): Unit = {
    protocolStatistics.updateBytesShipped(zeta.getSize)
    phi += zeta.getValue
    sync.shippedZetas += getCurrentCaller
    if (sync.shippedZetas.length == sync.parallelism) {
      sync.shippedZetas.clear()
      inc = 0
      if (phi >= barrier) {
        quantum = phi / (2.0 * sync.parallelism)
        newSubRound(quantum)
        sync.activeSubRound = true
      } else
        getBroadcastProxy.sendLocalDrift()
      phi = 0.0
    }
  }

  /** A method that broadcasts the new quantum to all the workers, thus initiating a new sub-round. */
  def newSubRound(quantum: Double): Unit = {
    require(quantum > 0)
    val q = Quantum(quantum)
    protocolStatistics.updateBytesShipped(sync.parallelism * q.getSize)
    protocolStatistics.asInstanceOf[FGMStatistics].updateNumOfSubRounds()
    getBroadcastProxy.receiveQuantum(q)
  }

  /** A marshalled model response.
   *
   * @return The marshalled model response.
   * */
  def sendModel(): Response[ParameterDescriptor] = {
    Response.respond(
      parametersDescription.copy(
        params = DenseVector.denseVectorConverter.convert(globalModel),
        fitted = fitted,
        miscellaneous = Quantum(quantum)
      )
    )
  }

  /**
   * The consumption method of user messages. Right know this is an empty method.
   *
   * @param data A data tuple for the Parameter Server.
   */
  @ProcessOp
  def receiveTuple[T <: Serializable](data: T): Unit = ()

  /** A method called when merging multiple Parameter Servers. Right know this is an empty method.
   *
   * @param parameterServers The parameter servers to merge this one with.
   * @return An array of [[FGMParameterServer]] instances.
   */
  @MergeOp
  def merge(parameterServers: Array[FGMParameterServer]): FGMParameterServer = {
    this
  }

  /** This method responds to a query for the Parameter Server. Right know this is an empty method.
   *
   * @param queryId     The query ID.
   * @param queryTarget The query target.
   * @param predicates  Any predicate that is necessary for the calculation of the query.
   */
  @QueryOp
  def query(queryId: Long, queryTarget: Int, predicates: Array[java.io.Serializable]): Unit = ()

  def setSafeZone(safeZone: SafeZone): Unit = this.safeZone = safeZone

  def setPrecision(precision: Double): Unit = this.precision = precision

  def getSafeZone: SafeZone = {
    val value = safeZone
    value
  }

  def getPrecision: Double = {
    val value = precision
    value
  }

  /** A helped local class for synchronizing the FGM protocol over the network. */
  class SyncProtocol {

    /** A variable indicating the number of workers of the distributed learning procedure. */
    var parallelism: Int = 0

    /** A flag variable determining the status of the current subround. */
    var activeSubRound: Boolean = true

    /** The workers that have sent their local drifts along with the number of data points that fitted to their local models. */
    var shippedDrifts: mutable.HashMap[Int, Long] = new mutable.HashMap[Int, Long]()

    /** The workers that have sent their local zetas. */
    var shippedZetas: ListBuffer[Int] = new ListBuffer[Int]()

  }

  override def configureParameterServer(request: Request): FGMParameterServer = {

    // Setting the ML Hub.
    val config: mutable.Map[String, AnyRef] = request.getTrainingConfiguration.asScala

    if (config.contains("precision")) {
      try {
        setPrecision(Parsing.DoubleParsing(config, "precision", 0.01))
      } catch {
        case e: Throwable => e.printStackTrace()
      }
    }

    if (config.contains("safeZone")) {
      try {
        setSafeZone(
          config("safeZone").asInstanceOf[String] match {
            case "ModelVariance" =>
              if (config.contains("threshold")) {
                try {
                  VarianceSafeZone(Parsing.DoubleParsing(config, "threshold", 0.0008))
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

  //  def checkParallelism(): Unit = {
  //    if (sync.parallelism != getNumberOfSpokes) {
  //      if (sync.parallelism > getNumberOfSpokes) {
  //        val difference: Int = sync.parallelism - getNumberOfSpokes
  //      } else {
  //        val difference: Int = getNumberOfSpokes - sync.parallelism
  //      }
  //    }
  //  }

}

