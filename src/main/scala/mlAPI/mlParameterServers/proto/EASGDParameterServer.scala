package mlAPI.mlParameterServers.proto

import BipartiteTopologyAPI.annotations.{InitOp, MergeOp, ProcessOp, QueryOp}
import ControlAPI.Request
import mlAPI.mlParameterServers.VectoredPS
import mlAPI.mlworkers.interfaces.Querier
import mlAPI.parameters.utils.ParameterDescriptor
import mlAPI.preprocessing.RunningMean
import mlAPI.protocols.periodic.{PullPush, RemoteLearner}
import mlAPI.protocols.statistics.EASGDStatistics
import mlAPI.utils.Parsing

import java.io.Serializable
import scala.collection.mutable
import scala.collection.JavaConverters._

case class EASGDParameterServer() extends VectoredPS[RemoteLearner, Querier] with PullPush {

  println("Elastic Averaging Hub initialized.")

  protocolStatistics = EASGDStatistics()

  /** The moving rate hyper-parameter. */
  protected var alpha: Double = _

  /** Initialization method of the Parameter server node. */
  @InitOp
  def init(): Unit = {
    parallelism = getNumberOfSpokes
    alpha = math.pow(0.9, 8) / getNumberOfSpokes
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

  /** An asynchronous update of the global model.
   *
   * @param mDesc The serialized model updates to be added to the global model.
   * */
  def updateGlobalState(mDesc: ParameterDescriptor): Boolean = {
    val updated: Boolean = updateParameterTree(mDesc)
    if (updated)
      try {
        globalVectorSlice += alpha * reconstructedVectorSlice
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
   * @return An array of [[EASGDParameterServer]] instances.
   */
  @MergeOp
  def merge(parameterServers: Array[EASGDParameterServer]): EASGDParameterServer = this

  /** This method responds to a query for the Parameter Server. Right know this is an empty method.
   *
   * @param queryId     The query ID.
   * @param queryTarget The query target.
   * @param predicates  Any predicate that is necessary for the calculation of the query.
   */
  @QueryOp
  def query(queryId: Long, queryTarget: Int, predicates: Array[java.io.Serializable]): Unit = ()

  def setAlpha(alpha: Double): Unit = this.alpha = alpha

  override def configureParameterServer(request: Request): EASGDParameterServer = {

    // Setting the ML node parameters.
    val config: mutable.Map[String, AnyRef] = request.getTrainingConfiguration.asScala

    // Set the s parameter.
    if (config.contains("alpha"))
      setAlpha(Parsing.DoubleParsing(config, "alpha", math.pow(0.9, 8) / getNumberOfSpokes))

    this
  }

}
