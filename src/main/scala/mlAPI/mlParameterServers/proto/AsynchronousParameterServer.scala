package mlAPI.mlParameterServers.proto

import BipartiteTopologyAPI.annotations.{InitOp, MergeOp, ProcessOp, QueryOp}
import BipartiteTopologyAPI.futures.Response
import mlAPI.mlParameterServers.VectoredPS
import mlAPI.mlworkers.interfaces.Querier
import mlAPI.parameters.utils.ParameterDescriptor
import mlAPI.protocols.periodic.{PushPull, RemoteLearner}
import mlAPI.protocols.statistics.AsynchronousStatistics

import java.io.Serializable

/**
 * The parameter server of an Asynchronous Distributed Learning Protocol.
 */
case class AsynchronousParameterServer() extends VectoredPS[RemoteLearner, Querier] with PushPull {

  println("Asynchronous Hub initialized.")

  protocolStatistics = AsynchronousStatistics()

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
   * A method used by the workers for pushing their models to the parameter server(s).
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
      }
    }
  }

  /** Asynchronously updating the local model and sending it back to the worker.
   *
   * @param mDesc The serialized model send from the worker.
   * @return The new global model that is sent back to the worker that pushed its last model updates.
   * */
  override def pushPull(mDesc: ParameterDescriptor): Response[ParameterDescriptor] = {
    protocolStatistics.updateBytesShipped(mDesc.getSize)
    if (updateGlobalState(mDesc)) {
      protocolStatistics.updateNumOfBlocks()
      protocolStatistics.updateModelsShipped()
      makePromise[ParameterDescriptor]()
      sendModelToWorker()
    } else
      makePromise[ParameterDescriptor]()
  }

  /** An asynchronous update of the global model.
   *
   * @param mDesc The serialized model updates to be added to the global model.
   * */
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
  def merge(parameterServers: Array[AsynchronousParameterServer]): AsynchronousParameterServer = this

  /** This method responds to a query for the Parameter Server. Right know this is an empty method.
   *
   * @param queryId     The query ID.
   * @param queryTarget The query target.
   * @param predicates  Any predicate that is necessary for the calculation of the query.
   */
  @QueryOp
  def query(queryId: Long, queryTarget: Int, predicates: Array[java.io.Serializable]): Unit = ()

}
