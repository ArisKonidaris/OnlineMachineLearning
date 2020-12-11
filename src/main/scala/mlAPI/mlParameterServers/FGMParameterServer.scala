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
import mlAPI.protocols.fgm.{FGMHubInterface, FGMRemoteLearner, Increment, Quantum, ZetaValue}
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
                              private var safeZone: SafeZone = VarianceSafeZone(0.0008))
  extends VectoredPS[FGMRemoteLearner, Querier] with FGMHubInterface {

  println("FGM Hub initialized.")

  /** A synchronizer variable. */
  var sync: SyncProtocol = new SyncProtocol()

  /** The increment of a sub-round. */
  var inc: Long = 0

  /** The number of rounds of the FMG distributed learning protocol. */
  var numRounds: Long = 0

  /** The number of sub-rounds of the FMG distributed learning protocol. */
  var numSubRounds: Long = 0

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
    numRounds += 1
    numSubRounds += 1
    safeZonesSent += sync.parallelism
  }

  /** Starting a new round of the FGM distributed learning protocol.
   *
   * @return The new global model and quantum for the new round.
   * */
  def startRound(): BroadcastValueResponse[ParameterDescriptor] = {
    prepareNewRound()
    incrementNumberOfShippedModels(sync.parallelism)
    fulfillBroadcastPromise(sendModel().getValue)
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
    parametersDescription = modelDescriptor
    globalModel = deserializeVector(modelDescriptor)
    drift = 0.0 * globalModel
    makeBroadcastPromise(new PromiseResponse[ParameterDescriptor]())
    sync.shippedDrifts.put(0, modelDescriptor.getFitted)
    incrementNumberOfReceivedModels()
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
    incrementNumberOfReceivedModels()
    drift += remoteVector
  }

  /** Receiving the drift of a worker.
   *
   * @param modelDescriptor The worker's drift from the global model.
   * @return The new global model.
   * */
  override def receiveLocalDrift(modelDescriptor: ParameterDescriptor): Response[ParameterDescriptor] = {
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
    if (sync.activeSubRound && numSubRounds == increment.subRound) {
      inc += increment.increment
      if (inc > sync.parallelism) {
//        phi = 0.0
        sync.activeSubRound = false
        for (workerID: Int <- 0 until sync.parallelism)
          getProxy(workerID).requestZeta()
      }
    }
//    println(inc)
//    printStatistics()
  }

  /** Receiving the zeta safe zone function value of a worker.
   *
   * @param zeta The zeta safezone function value sent by the worker.
   * */
  override def receiveZeta(zeta: ZetaValue): Unit = {
    phi += zeta.getValue
    sync.shippedZetas += getCurrentCaller
    if (sync.shippedZetas.length == sync.parallelism) {
      sync.shippedZetas.clear()
      inc = 0
      if (phi >= barrier) {
        quantum = phi / (2.0 * sync.parallelism)
        assert(quantum > 0)
        for (workerID: Int <- 0 until sync.parallelism)
          getProxy(workerID).receiveQuantum(Quantum(quantum))
        numSubRounds += 1
        sync.activeSubRound = true
      } else {
        for (workerID: Int <- 0 until sync.parallelism)
          getProxy(workerID).sendLocalDrift()
      }
      phi = 0.0
    }
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
        miscellaneous = Array(quantum)
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

  def configure(request: Request): Unit = {

    // Setting the ML Hub.
    val config: mutable.Map[String, AnyRef] = request.getTraining_configuration.asScala

    if (config.contains("precision")) {
      try {
        setPrecision(Parsing.DoubleParsing(config, "precision", 0.01))
      } catch {
        case e: Throwable => e.printStackTrace()
      }
    }

    if (config.contains("safe_zone")) {
      try {
        setSafeZone(
          config("safe_zone").asInstanceOf[String] match {
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

