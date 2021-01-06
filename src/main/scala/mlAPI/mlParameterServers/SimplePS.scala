package mlAPI.mlParameterServers

import BipartiteTopologyAPI.annotations.{InitOp, MergeOp, ProcessOp, QueryOp}
import BipartiteTopologyAPI.futures.Response
import mlAPI.mlworkers.interfaces.Querier
import mlAPI.parameters.ParameterDescriptor
import mlAPI.protocols.periodic.{Push, RemoteLearner}
import mlAPI.math.DenseVector
import mlAPI.protocols.periodic.simple.SimpleStatistics

import java.io.Serializable

case class SimplePS() extends VectoredPS[RemoteLearner, Querier] with Push {

  println("Simple Hub initialized.")

  protocolStatistics = SimpleStatistics()

  /** Initialization method of the Parameter server node. */
  @InitOp
  def init(): Unit = ()

  /** The consumption method of user messages. Right know this is an empty method.
   *
   * @param data A data tuple for the Parameter Server.
   */
  @ProcessOp
  def receiveTuple[T <: Serializable](data: T): Unit = ()

  /** A method called when merging multiple Parameter Servers. Right know this is an empty method.
   *
   * @param parameterServers The parameter servers to merge this one with.
   * @return An array of [[SimplePS]] instances.
   */
  @MergeOp
  def merge(parameterServers: Array[SimplePS]): SimplePS = this

  /** This method responds to a query for the Parameter Server. Right know this is an empty method.
   *
   * @param queryId     The query ID.
   * @param queryTarget The query target.
   * @param predicates  Any predicate that is necessary for the calculation of the query.
   */
  @QueryOp
  def query(queryId: Long, queryTarget: Int, predicates: Array[java.io.Serializable]): Unit = ()

  /** An asynchronous update of the global model.
   *
   * @param remoteModelDescriptor The serialized model updates to be added to the global model.
   * */
  def updateModel(remoteModelDescriptor: ParameterDescriptor): Unit = {
    incrementNumberOfFittedData(remoteModelDescriptor.getFitted)
    globalModel = deserializeVector(remoteModelDescriptor)
    parametersDescription = remoteModelDescriptor
  }

  override def pushModel(modelDescriptor: ParameterDescriptor): Unit = {
    protocolStatistics.updateBytesShipped(modelDescriptor.getSize)
    protocolStatistics.updateModelsShipped()
    updateModel(modelDescriptor)
  }

}
