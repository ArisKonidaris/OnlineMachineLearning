package mlAPI.mlParameterServers.proto

import BipartiteTopologyAPI.annotations.{InitOp, MergeOp, ProcessOp, QueryOp}
import BipartiteTopologyAPI.futures.Response
import mlAPI.mlParameterServers.VectoredPS
import mlAPI.mlworkers.interfaces.Querier
import mlAPI.parameters.utils.ParameterDescriptor
import mlAPI.protocols.dynamic.{GMHubInterface, GMRemoteLearner}
import mlAPI.protocols.statistics.GMStatistics
import breeze.linalg.{DenseVector => BreezeDenseVector}

import java.io.Serializable
import scala.collection.JavaConverters._

case class GMParameterServer() extends VectoredPS[GMRemoteLearner, Querier] with GMHubInterface {

  println("GM Hub initialized.")

  protocolStatistics = GMStatistics()

  /** A helping counter. */
  private var counter: Int = 0

  /** A breeze vector. */
  private var vector: BreezeDenseVector[Double] = _

  /** A flag determining if the hub is in the process of obtaining all local models in order to start a new GM round. */
  private var pendingNewRound: Boolean = false

  /** Initialization method of the parameter server node. */
  @InitOp
  def init(): Unit = {
    parallelism = getNumberOfSpokes
    protocolStatistics.setProtocol(
      protocolStatistics.getProtocol + s"($getNumberOfSpokes,$getNumberOfHubs)"
    )
  }

  /** Ending the warmup of the GM distributed learning protocol.
   *
   * @param mDesc The marshalled model sent by the first worker for warming up the GM distributed
   *              learning procedure.
   * @return The global model to be broadcasted to all the workers in order to start the GM distributed
   *         learning procedure.
   * */
  override def endWarmup(mDesc: ParameterDescriptor): Unit = {
    protocolStatistics.updateBytesShipped(mDesc.getSize)
    if (updateParameterTree(mDesc)) {
      protocolStatistics.updateModelsShipped()
      protocolStatistics.updateNumOfBlocks()
      globalVectorSlice = reconstructedVectorSlice.copy
      reconstructedVectorSlice = null
      vector = 0.0 * globalVectorSlice
      warmupCounter += 1
      if (warmupCounter == parallelism)
        warmWorkers()
    }
  }

  /** A method invoked when a worker as soon it is initialized to request the warmed up global model. */
  override def pull(): Unit = {
    assert(!isWarmedUp)
    protocolStatistics.updateNumOfBlocks()
    warmupCounter += 1
    if (warmupCounter == parallelism)
      warmWorkers()
  }

  /** A method for warming up the workers. */
  def warmWorkers(): Unit = {
    val model = warmUpModel()
    protocolStatistics.updateModelsShipped(parallelism - 1)
    protocolStatistics.updateBytesShipped((parallelism - 1) * (for (slice <- model) yield slice.getSize).sum)
    if (getNodeId == 0) {
      assert(roundLoss.getCount == 1)
      updateLearningCurve()
    }
    for (worker: Int <- 1 until parallelism)
      for (slice <- model)
        getProxy(worker).updateModel(slice)
    if (getNodeId == 0)
      getProxy(0).updateModel(ParameterDescriptor(null, null, null, null, null, null))
  }

  /** THis method is called by a worker when its admissible regin has been violated. */
  override def violation(): Unit = {
    if (!pendingNewRound) {
      println("---> Violation " + getCurrentCaller)
      pendingNewRound = true
      getBroadcastProxy.sendLocalModel()
    }
  }

  /** Receiving the model of a worker.
   *
   * @param mDesc The worker's model.
   * @return The new global model.
   */
  override def receiveLocalModel(mDesc: ParameterDescriptor): Response[ParameterDescriptor] = {
    protocolStatistics.updateBytesShipped(mDesc.getSize)
    if (updateParameterTree(mDesc)) {
      protocolStatistics.updateModelsShipped()
      protocolStatistics.updateNumOfBlocks()
      counter += 1
      vector += reconstructedVectorSlice
      reconstructedVectorSlice = null
      if (counter == parallelism) {
        globalVectorSlice = (1.0 / (1.0 * counter)) * vector
        if (getNodeId == 0) {
          assert(roundLoss.getCount == counter)
          updateLearningCurve()
        }
        makeBroadcastPromise[ParameterDescriptor]()
        startRound()
      } else
        makeBroadcastPromise[ParameterDescriptor]()
    } else
      makeBroadcastPromise[ParameterDescriptor]()
  }

  /** Starting a new round of the GM distributed learning protocol.
    *
    * @return The new global model for the new round.
    */
  def startRound(): Response[ParameterDescriptor] = {
    vector *= 0.0
    pendingNewRound = false
    counter = 0
    val model = serializableModel()
    protocolStatistics.asInstanceOf[GMStatistics].updateNumOfRounds()
    protocolStatistics.updateBytesShipped(parallelism * (for (slice <- model) yield slice.getSize).sum)
    protocolStatistics.updateModelsShipped(parallelism)
    fulfillBroadcastPromises(model.toList.asJava)
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
   * @return An array of [[GMParameterServer]] instances.
   */
  @MergeOp
  def merge(parameterServers: Array[GMParameterServer]): GMParameterServer = {
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

}
