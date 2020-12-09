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
 * A Parameter Server for Synchronous Distributed Learning with a global vector model.
 */
case class SynchronousParameterServer() extends VectoredPS[RemoteLearner, Querier] with PullPush {

  println("Synchronous Hub initialized.")

  /** A helping counter. */
  var counter: Int = 0

  /** A flag indicating if the FGM network is warmed up. */
  var warmedUp: Boolean = false

  /** A variable indicating the number of workers of the distributed learning procedure. */
  var parallelism: Int = 0

  /** Initialization method of the Parameter server node. */
  @InitOp
  def init(): Unit = parallelism = getNumberOfSpokes

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
   * @return An array of [[SynchronousParameterServer]] instances.
   */
  @MergeOp
  def merge(parameterServers: Array[SynchronousParameterServer]): SynchronousParameterServer = {
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

  /** Pulling the global model.
   *
   * @return The global model to be send back to worker that requested the pull.
   * */
  override def pullModel: Response[ParameterDescriptor] = {
    assert(getCurrentCaller != 0)
    makeBroadcastPromise(new PromiseResponse[ParameterDescriptor]())
    counter += 1
    if (globalModel == null)
      Response.noResponse()
    else if (counter == parallelism) {
      counter = 0
      warmedUp = true
      incrementNumberOfShippedModels(parallelism)
      fulfillBroadcastPromise(sendModel().getValue)
    } else
      Response.noResponse()
  }

  /** Synchronously updating the local model.
   *
   * @param modelDescriptor The serialized model send from the worker.
   * @return The new global model that is sent back to all workers when all the updates have been gathered.
   * */
  override def pushModel(modelDescriptor: ParameterDescriptor): Response[ParameterDescriptor] = {
    updateGlobalState(modelDescriptor)
    if (warmedUp) {
      makeBroadcastPromise(new PromiseResponse[ParameterDescriptor]())
      counter += 1
      if (counter == getNumberOfSpokes) {
        counter = 0
        incrementNumberOfShippedModels(parallelism)
        fulfillBroadcastPromise(sendModel().getValue)
      } else
        Response.noResponse()
    } else if (counter == parallelism) {
      warmedUp = true
      counter = 0
      incrementNumberOfShippedModels(parallelism)
      fulfillBroadcastPromise(sendModel().getValue)
    } else
      Response.noResponse()
  }

  /** An synchronous update of the global model.
   *
   * @param remoteModelDescriptor The serialized model updates to be added to the global model.
   * */
  def updateGlobalState(remoteModelDescriptor: ParameterDescriptor): Unit = {
    val remoteVector: BreezeDenseVector[Double] = deserializeVector(remoteModelDescriptor)
    incrementNumberOfFittedData(remoteModelDescriptor.getFitted)
    incrementNumberOfReceivedModels()
//    printStatistics()
    if (globalModel == null) {
      assertWarmup(remoteModelDescriptor)
      globalModel = remoteVector
      parametersDescription = remoteModelDescriptor
      makeBroadcastPromise(new PromiseResponse[ParameterDescriptor]())
      counter += 1
    } else {
      if (counter == 0)
        globalModel = remoteVector
      else
        globalModel += remoteVector
    }
  }

  /** A marshalled model response.
   *
   * @return The marshalled model response.
   * */
  def sendModel(): Response[ParameterDescriptor] = {
    Response.respond(
      parametersDescription.copy(
        params = DenseVector.denseVectorConverter.convert(globalModel * (1.0 / (1.0 * getNumberOfSpokes))),
        fitted = fitted
      )
    )
  }

}
