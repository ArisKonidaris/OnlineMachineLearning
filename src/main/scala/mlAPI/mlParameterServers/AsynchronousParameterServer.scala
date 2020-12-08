package mlAPI.mlParameterServers

import java.io.Serializable
import BipartiteTopologyAPI.annotations.{InitOp, MergeOp, ProcessOp, QueryOp}
import BipartiteTopologyAPI.futures.{PromiseResponse, Response}
import mlAPI.math.DenseVector
import mlAPI.mlworkers.interfaces.Querier
import mlAPI.parameters.ParameterDescriptor
import breeze.linalg.{DenseVector => BreezeDenseVector}
import mlAPI.protocols.periodic.{PullPush, RemoteLearner}

/**
 * The parameter server of an Asynchronous Distributed Learning Protocol.
 */
case class AsynchronousParameterServer() extends VectoredPS[RemoteLearner, Querier] with PullPush {

  println("Asynchronous Hub initialized.")

  /** A helping counter. */
  var warmupCounter: Int = 0

  /** A flag indicating if the FGM network is warmed up. */
  var warmedUp: Boolean = false

  /** A variable indicating the number of workers of the distributed learning procedure. */
  var parallelism: Int = _

  /** Initialization method of the Parameter server node. */
  @InitOp
  def init(): Unit = parallelism = getNumberOfSpokes

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
  def merge(parameterServers: Array[AsynchronousParameterServer]): AsynchronousParameterServer = {
    this
  }

  /** This method responds to a query for the Parameter Server. Right know this is an empty method.
   *
   * @param queryId     The query ID.
   * @param queryTarget The query target.
   * @param predicates  Any predicate that is necessary for the calculation of the query.
   */
  @QueryOp
  def query(queryId: Long, queryTarget: Int, predicates: Array[java.io.Serializable]): Unit = {
  }

  /** Pulling the global model.
   *
   * @return The global model to be send back to worker that requested the pull.
   * */
  override def pullModel: Response[ParameterDescriptor] = {
    assert(getCurrentCaller != 0)
    makeBroadcastPromise(new PromiseResponse[ParameterDescriptor]())
    warmupCounter += 1
    if (globalModel == null)
      Response.noResponse()
    else if (warmupCounter == parallelism) {
      warmupCounter = 0
      warmedUp = true
      incrementNumberOfShippedModels(parallelism)
      fulfillBroadcastPromise(sendModel().getValue)
    } else
      Response.noResponse()
  }

  /** Asynchronously updating the local model and sending it back to the worker.
   *
   * @param modelDescriptor The serialized model send from the worker.
   * @return The new global model that is sent back to the worker that pushed its last model updates.
   * */
  override def pushModel(modelDescriptor: ParameterDescriptor): Response[ParameterDescriptor] = {
    updateGlobalState(modelDescriptor)
    if (warmedUp) {
      incrementNumberOfShippedModels()
      sendModel()
    } else if (warmupCounter == parallelism) {
      warmedUp = true
      warmupCounter = 0
      incrementNumberOfShippedModels(parallelism)
      fulfillBroadcastPromise(sendModel().getValue)
    } else {
      Response.noResponse()
    }
  }

  /** An asynchronous update of the global model.
   *
   * @param remoteModelDescriptor The serialized model updates to be added to the global model.
   * */
  def updateGlobalState(remoteModelDescriptor: ParameterDescriptor): Unit = {
    val remoteVector: BreezeDenseVector[Double] = deserializeVector(remoteModelDescriptor)
    incrementNumberOfFittedData(remoteModelDescriptor.getFitted)
    incrementNumberOfReceivedModels()
    printStatistics()
    try {
      globalModel += (remoteVector * (1.0 / (1.0 * getNumberOfSpokes)))
    } catch {
      case _: Throwable =>
        assertWarmup(remoteModelDescriptor)
        globalModel = remoteVector
        parametersDescription = remoteModelDescriptor
        makeBroadcastPromise(new PromiseResponse[ParameterDescriptor]())
        warmupCounter += 1
    }
  }

  /** A marshalled model response.
   *
   * @return The marshalled model response.
   * */
  def sendModel(): Response[ParameterDescriptor] = {
    Response.respond(
      parametersDescription.copy(params = DenseVector.denseVectorConverter.convert(globalModel), fitted = fitted)
    )
  }

}
