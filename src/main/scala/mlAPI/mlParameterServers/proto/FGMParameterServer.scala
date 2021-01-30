package mlAPI.mlParameterServers.proto

import BipartiteTopologyAPI.annotations.{InitOp, MergeOp, ProcessOp, QueryOp}
import BipartiteTopologyAPI.futures.Response
import ControlAPI.Request
import mlAPI.mlParameterServers.VectoredPS
import mlAPI.mlworkers.interfaces.Querier
import mlAPI.parameters.utils.ParameterDescriptor
import mlAPI.protocols.dynamic._
import mlAPI.protocols.statistics.FGMStatistics
import mlAPI.safezones.{SafeZone, VarianceSafeZone}
import mlAPI.utils.Parsing
import breeze.linalg.{DenseVector => BreezeDenseVector}

import java.io.Serializable
import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.collection.JavaConverters._

case class FGMParameterServer(private var precision: Double = 0.01,
                              private var safeZone: SafeZone = VarianceSafeZone())
  extends VectoredPS[FGMRemoteLearner, Querier] with FGMHubInterface {

  println("FGM Hub initialized.")

  protocolStatistics = FGMStatistics()

  /** A flag variable determining the status of the current subround. */
  var activeSubRound: Boolean = true

  /** The workers that have sent their local drifts along with a number indicating if they have been updating their
   * local models since the last synchronization. */
  var shippedDrifts: mutable.HashMap[Int, Int] = new mutable.HashMap[Int, Int]()

  /** The workers that have sent their local zetas. */
  var shippedZetas: ListBuffer[Int] = new ListBuffer[Int]()

  /** The increment of a sub-round. */
  var inc: Long = 0L

  /** Number of safe zones sent by the FGM coordinator. */
  var safeZonesSent: Long = 0L

  /** The phi value of the FMG distributed learning protocol. */
  var phi: Double = 0D

  /** The quantum of the FMG distributed learning protocol. */
  var quantum: Double = 0D

  /** The smallest number the zeta function can reach. */
  var barrier: Double = 0D

  /** The drift vector. */
  var drift: BreezeDenseVector[Double] = _

  /** Initialization method of the Parameter server node. */
  @InitOp
  def init(): Unit = {
    parallelism = getNumberOfSpokes
    protocolStatistics.setProtocol(
      protocolStatistics.getProtocol + s"($getNumberOfSpokes,$getNumberOfHubs)"
    )
  }

  /** Ending the warmup of the FGM distributed learning protocol.
   *
   * @param mDesc The marshalled model sent by the first worker for warming up the FGM distributed
   *              learning procedure.
   * @return The global model to be broadcasted to all the workers in order to start the FGM distributed
   *         learning procedure.
   * */
  override def endWarmup(mDesc: ParameterDescriptor): Unit = {
    protocolStatistics.updateBytesShipped(mDesc.getSize)
    if (updateParameterTree(mDesc)) {
      protocolStatistics.updateModelsShipped()
      protocolStatistics.updateNumOfBlocks()
      globalVectorSlice = reconstructedVectorSlice.copy
      reconstructedVectorSlice = null
      drift = 0.0 * globalVectorSlice
      shippedDrifts.put(0, 1)
      if (shippedDrifts.size == parallelism)
        warmWorkers()
    }
  }

  /** A method invoked when a worker as soon it is initialized to request the warmed up global model. */
  override def pull(): Unit = {
    assert(!isWarmedUp)
    protocolStatistics.updateNumOfBlocks()
    shippedDrifts.put(getCurrentCaller, 0)
    if (shippedDrifts.size == parallelism)
      warmWorkers()
  }

  /** A method for warming up the workers. */
  def warmWorkers(): Unit = {
    prepareNewRound()
    val model = warmUpModel()
    if (model.last.getMiscellaneous != null)
      model.last.setMiscellaneous(model.last.getMiscellaneous ++ Array(Quantum(quantum)))
    else
      model.last.setMiscellaneous(Array(Quantum(quantum)))
    protocolStatistics.updateModelsShipped(parallelism - 1)
    protocolStatistics.updateBytesShipped((parallelism - 1) * (for (slice <- model) yield slice.getSize).sum)
    for (worker: Int <- 1 until parallelism)
      for (slice <- model)
        getProxy(worker).newRound(slice)
    protocolStatistics.updateBytesShipped(Quantum(quantum).getSize)
    getProxy(0).newRound(ParameterDescriptor(null, null, null, null, Array(Quantum(quantum)), null))
  }

  /** This method prepares the FGM network for a new round. */
  def prepareNewRound(): Unit = {

    // Resets
    shippedDrifts.clear()
    drift *= 0.0
    activeSubRound = true

    // Calculating the new phi, quantum and the minimum acceptable value for phi.
    phi = parallelism * safeZone.newRoundZeta()
    quantum = phi / (2.0 * parallelism)
    barrier = precision * phi

    // Increment the number of rounds, sub-rounds and number of shipped safe zones of the FGM distributed learning protocol.
    protocolStatistics.asInstanceOf[FGMStatistics].updateNumOfRounds()
    protocolStatistics.asInstanceOf[FGMStatistics].updateNumOfSubRounds()
    safeZonesSent += parallelism
  }

  /** Starting a new round of the FGM distributed learning protocol.
   *
   * @return The new global model and quantum for the new round.
   */
  def startRound(): Response[ParameterDescriptor] = {
    prepareNewRound()
    val model = serializableModel()
    model.last.setMiscellaneous(Array(Quantum(quantum)))
    protocolStatistics.updateBytesShipped(parallelism * (for (slice <- model) yield slice.getSize).sum)
    protocolStatistics.updateModelsShipped(parallelism)
    fulfillBroadcastPromises(model.toList.asJava)
  }

  /** Receiving an increment of a worker.
   *
   * @param increment The increment send by the worker.
   * */
  override def receiveIncrement(increment: Increment): Unit = {
    protocolStatistics.updateBytesShipped(increment.getSize)
    if (activeSubRound && protocolStatistics.asInstanceOf[FGMStatistics].getNumOfSubRounds == increment.subRound) {
      inc += increment.getIncrement
      if (inc > parallelism) {
        phi = 0.0
        activeSubRound = false
        println("-> BROADCAST REQUEST ZETA")
        getBroadcastProxy.requestZeta()
//        for (worker <- 0 until parallelism)
//          getProxy(worker).requestZeta()
      }
    }
  }

  /** Receiving the zeta safe zone function value of a worker.
   *
   * @param zeta The zeta safe zone function value sent by the worker.
   * */
  override def receiveZeta(zeta: ZetaValue): Unit = {
    protocolStatistics.updateBytesShipped(zeta.getSize)
    phi += zeta.getValue
    shippedZetas += getCurrentCaller
    if (shippedZetas.length == parallelism) {
      shippedZetas.clear()
      inc = 0
      if (phi > barrier) {
        quantum = phi / (2.0 * parallelism)
        newSubRound(quantum)
        activeSubRound = true
      } else
        getBroadcastProxy.sendLocalDrift()
    }
  }

  /** A method that broadcasts the new quantum to all the workers, thus initiating a new sub-round. */
  def newSubRound(quantum: Double): Unit = {
    require(quantum > 0)
    val q = Quantum(quantum)
    protocolStatistics.updateBytesShipped(parallelism * q.getSize)
    protocolStatistics.asInstanceOf[FGMStatistics].updateNumOfSubRounds()
    getBroadcastProxy.receiveQuantum(q)
  }

  /** Receiving the drift of a worker.
   *
   * @param mDesc The worker's drift from the global model.
   * @return The new global model.
   */
  override def receiveLocalDrift(mDesc: ParameterDescriptor): Response[ParameterDescriptor] = {
    protocolStatistics.updateBytesShipped(mDesc.getSize)
    if (updateParameterTree(mDesc)) {
      protocolStatistics.updateModelsShipped()
      protocolStatistics.updateNumOfBlocks()
      shippedDrifts.put(getCurrentCaller, 1)
      drift += reconstructedVectorSlice
      if (shippedDrifts.size == parallelism) {
        globalVectorSlice += (1.0 / (1.0 * shippedDrifts.toArray.map(x => x._2).sum)) * drift
        makeBroadcastPromise[ParameterDescriptor]()
        startRound()
      } else
        makeBroadcastPromise[ParameterDescriptor]()
    } else
      makeBroadcastPromise[ParameterDescriptor]()
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

  def getSafeZone: SafeZone = safeZone

  def getPrecision: Double = precision

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

}
