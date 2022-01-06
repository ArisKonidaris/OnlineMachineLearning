package mlAPI.mlParameterServers.proto

import BipartiteTopologyAPI.annotations.{InitOp, MergeOp, ProcessOp, QueryOp}
import BipartiteTopologyAPI.futures.Response
import ControlAPI.Request
import mlAPI.mlParameterServers.VectoredPS
import mlAPI.mlworkers.interfaces.Querier
import mlAPI.parameters.utils.ParameterDescriptor
import mlAPI.protocols.LongWrapper
import mlAPI.protocols.periodic.{PushPull, RemoteLearner}
import mlAPI.protocols.statistics.StaleSynchronousStatistics
import mlAPI.utils.Parsing

import java.io.Serializable
import scala.collection.mutable
import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer

case class SSPParameterServer() extends VectoredPS[RemoteLearner, Querier] with PushPull {

  println("Stale Synchronous Hub initialized.")

  protocolStatistics = StaleSynchronousStatistics()

  /** The staleness parameter. */
  var s: Int = 3

  /** The worker clocks. */
  val clocks: mutable.Map[Int, Long] = mutable.HashMap[Int, Long]()

  /** The minimum worker clock. */
  var min: Long = 0

  /** The pending promises. */
  var promises: mutable.Map[Int, Long] = mutable.HashMap[Int, Long]()

  /** Initialization method of the Parameter server node. */
  @InitOp
  def init(): Unit = {
    parallelism = getNumberOfSpokes
    protocolStatistics.setProtocol(
      protocolStatistics.getProtocol + s"($getNumberOfSpokes,$getNumberOfHubs)"
    )
  }

  /** A method for warming up the workers. */
  def warmWorkers(): Unit = {
    val model = warmUpModel()
    protocolStatistics.updateModelsShipped(parallelism - 1)
    protocolStatistics.updateBytesShipped((parallelism - 1) * (for (slice <- model) yield slice.getSize).sum)
    for (worker: Int <- 1 until parallelism)
      for (slice <- model)
        getProxy(worker).updateModel(slice)
    getProxy(0).updateModel(ParameterDescriptor(null, null, null, null, null, null))
  }

  /** A method used by the workers for requesting the global model from the parameter server(s). */
  override def pull(): Unit = {
    protocolStatistics.updateNumOfBlocks()
    if (isWarmedUp) {
      protocolStatistics.updateModelsShipped()
      for (slice <- serializableModel()) {
        protocolStatistics.updateBytesShipped(slice.getSize)
        getProxy(getCurrentCaller).updateModel(slice)
      }
    } else {
      warmupCounter += 1
      if (warmupCounter == parallelism)
        warmWorkers()
    }
  }

  /**
   * A method used by the workers for pushing their models to the parameter server(s) along with their local clock.
   *
   * @param mDesc The model pushed by the worker.
   */
  override def push(mDesc: ParameterDescriptor): Unit = {
    protocolStatistics.updateBytesShipped(mDesc.getSize)
    if (updateGlobalState(mDesc)) {
      protocolStatistics.updateModelsShipped()
      if (!isWarmedUp) {
        warmupCounter += 1
        if (warmupCounter == parallelism)
          warmWorkers()
      } else
        updateClocks(mDesc.getMiscellaneous.head.asInstanceOf[LongWrapper].getLong)
    }
  }

  /** Asynchronously updating the local model and sending it back to the worker.
   *
   * @param mDesc The serialized model send from the worker.
   * @return The new global model that is sent back to the worker that pushed its last model updates.
   */
  override def pushPull(mDesc: ParameterDescriptor): Response[ParameterDescriptor] = {
    protocolStatistics.updateBytesShipped(mDesc.getSize)
    if (updateGlobalState(mDesc)) {
      protocolStatistics.updateNumOfBlocks()
      protocolStatistics.updateModelsShipped()
      val callerClock: Long = mDesc.getMiscellaneous.head.asInstanceOf[LongWrapper].getLong
      updateClocks(callerClock)
      if (callerClock - min <= s) {
        makePromise[ParameterDescriptor]()
        sendModelToWorker()
      } else {
        promises.put(getCurrentCaller, callerClock)
        assert(promises.values.max - promises.values.min <= s + 1)
        makePromise[ParameterDescriptor]()
      }
    } else
      makePromise[ParameterDescriptor]()
  }

  /** An asynchronous update of the global model.
   *
   * @param mDesc The serialized model updates to be added to the global model.
   */
  def updateGlobalState(mDesc: ParameterDescriptor): Boolean = {
    val updated: Boolean = updateParameterTree(mDesc)
    if (updated)
      try {
        globalVectorSlice += (reconstructedVectorSlice * (1.0 / (1.0 * getNumberOfSpokes)))
        if (getNodeId == 0 && roundLoss.getCount == parallelism)
          updateLearningCurve()
      } catch {
        case _: Throwable =>
          globalVectorSlice = reconstructedVectorSlice.copy
          if (getNodeId == 0) {
            assert(roundLoss.getCount == 1)
            updateLearningCurve()
          }
      } finally {
        reconstructedVectorSlice = null
      }
    updated
  }

  /** Updating the worker clocks. */
  def updateClocks(clock: Long): Unit = {
    if (!clocks.contains(getCurrentCaller))
      clocks.put(getCurrentCaller, clock)
    else
      clocks(getCurrentCaller) = clock
    min = clocks.values.min
    assert(min >= 0)
    checkPromises()
  }

  /** Send the new model to the fast workers. */
  def checkPromises(): Unit = {
    if (promises.nonEmpty) {
      val replied = ListBuffer[Int]()
      var model: Array[ParameterDescriptor] = null
      for (promise <- promises)
        if (promise._2 - min <= s) {
          if (model == null) {
            model = serializableModel()
            model.last.setMiscellaneous(Array(LongWrapper(min)))
          }
          protocolStatistics.updateModelsShipped()
          protocolStatistics.updateBytesShipped((for (slice <- model) yield slice.getSize).sum)
          fulfillPromises(promise._1, model.toList.asJava).sendAnswers()
          replied += promise._1
        }
      for (reply <- replied)
        promises.remove(reply)
    }
  }

  /** The consumption method of user messages. Right know this is an empty method.
   *
   * @param data A data tuple for the Parameter Server.
   */
  @ProcessOp
  def receiveTuple[T <: Serializable](data: T): Unit = ()

  /** A method called when merging multiple Parameter Servers. Right know this is an empty method.
   *
   * @param parameterServers The parameter servers to merge this one with.
   * @return An array of [[AsynchronousParameterServer]] instances.
   */
  @MergeOp
  def merge(parameterServers: Array[SSPParameterServer]): SSPParameterServer = this

  /** This method responds to a query for the Parameter Server. Right know this is an empty method.
   *
   * @param queryId     The query ID.
   * @param queryTarget The query target.
   * @param predicates  Any predicate that is necessary for the calculation of the query.
   */
  @QueryOp
  def query(queryId: Long, queryTarget: Int, predicates: Array[java.io.Serializable]): Unit = ()

  override def configureParameterServer(request: Request): SSPParameterServer = {

    // Setting the ML node parameters.
    val config: mutable.Map[String, AnyRef] = request.getTrainingConfiguration.asScala

    // Set the s parameter.
    if (config.contains("s")) {
      s = {
        try {
          Parsing.IntegerParsing(config, "s", 3)
        } catch {
          case _: Throwable => 3
        }
      }
    }

    this
  }

}
