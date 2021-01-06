package mlAPI.mlworkers.worker.PeriodicWorkers

import BipartiteTopologyAPI.annotations.{InitOp, ProcessOp, QueryOp}
import ControlAPI.QueryResponse
import mlAPI.math.Point
import mlAPI.mlworkers.interfaces.Querier
import mlAPI.parameters.ParameterDescriptor
import mlAPI.protocols.periodic.{PullPush, RemoteLearner}

import scala.collection.mutable.ListBuffer
import scala.collection.JavaConverters._

case class SynchronousWorker() extends PeriodicVectoredWorker[PullPush, Querier] with RemoteLearner {

  println("Synchronous Worker initialized.")

  protocol = "Synchronous Protocol"

  /** Initialization method of the Machine Learning worker node. */
  @InitOp
  def init(): Unit = if (getNodeId != 0) pull()

  /** The consumption of a data point by the Machine Learning worker.
   *
   * @param data A data point to be fitted to the model.
   */
  @ProcessOp
  def receiveTuple(data: Point): Unit = {
    fit(data)
    if (processedData >= getMiniBatchSize * miniBatches) push()
  }

  /** Pushing the local model to the parameter server(s). */
  def push(): Unit = {
    for ((slice: ParameterDescriptor, index: Int) <- ModelMarshalling(drift = false).zipWithIndex)
      getProxy(index).pushModel(slice).toSync(updateModel)
    processedData = 0
  }

  /** Pulling the global model from the parameter server(s). */
  def pull(): Unit = {
    for (i <- 0 until getNumberOfHubs)
      getProxy(i).pullModel.toSync(updateModel)
  }

  /** This method responds to a query for the Machine Learning worker.
   *
   * @param predicates The predicated of the query.
   */
  @QueryOp
  def query(queryId: Long, queryTarget: Int, predicates: (Double, Array[Point])): Unit = {
    val pj = mlPipeline.generatePOJO
    val score = getGlobalPerformance(ListBuffer(predicates._2: _ *))
    if (queryId == -1)
      getQuerier.sendQueryResponse(
        new QueryResponse(-1,
          queryTarget,
          null,
          null,
          null,
          processedData,
          null,
          predicates._1,
          score)
      )
    else
      getQuerier.sendQueryResponse(
        new QueryResponse(queryId, queryTarget, pj._1.asJava, pj._2, protocol, pj._3, pj._4, pj._5, score)
      )
  }

}
