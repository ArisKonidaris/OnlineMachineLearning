package mlAPI.mlworkers.worker

import ControlAPI.QueryResponse
import BipartiteTopologyAPI.annotations.{InitOp, ProcessOp, QueryOp}
import mlAPI.math.{DenseVector, Point, SparseVector}
import mlAPI.mlworkers.interfaces.Querier
import mlAPI.parameters.ParameterDescriptor
import mlAPI.protocols.periodic.{PullPush, RemoteLearner}

import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer

case class SynchronousWorker() extends VectoredWorker[PullPush, Querier] with RemoteLearner {

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
    if (processedData >= miniBatchSize * miniBatches) push()
  }

  /** A method called each type the new global model
   * (or a slice of it) arrives from the parameter server.
   */
  override def updateModel(mDesc: ParameterDescriptor): Unit = {
    if (getNumberOfHubs == 1) {
      globalModel = mlPipeline.getLearner.generateParameters(mDesc)
      mlPipeline.getLearner.setParameters(globalModel.getCopy)
      mlPipeline.setFittedData(mDesc.getFitted)
      processedData = 0
    } else {
      parameterTree.put((mDesc.getBucket.getStart.toInt, mDesc.getBucket.getEnd.toInt), mDesc.getParams)
      if (mlPipeline.getFittedData < mDesc.getFitted) mlPipeline.setFittedData(mDesc.getFitted)
      if (parameterTree.size == getNumberOfHubs) {
        mDesc.setParams(
          DenseVector(
            parameterTree.values
              .map(
                {
                  case dense: DenseVector => dense
                  case sparse: SparseVector => sparse.toDenseVector
                })
              .fold(Array[Double]())(
                (accum, vector) => accum.asInstanceOf[Array[Double]] ++ vector.asInstanceOf[DenseVector].data)
              .asInstanceOf[Array[Double]]
          )
        )
        globalModel = mlPipeline.getLearner.generateParameters(mDesc)
        mlPipeline.getLearner.setParameters(globalModel.getCopy)
        processedData = 0
      }
    }
  }

  /** This method responds to a query for the Machine Learning worker.
   *
   * @param test_set The test set that the predictive performance of the model should be calculated on.
   */
  @QueryOp
  def query(queryId: Long, queryTarget: Int, test_set: Array[java.io.Serializable]): Unit = {
    val pj = mlPipeline.generatePOJO(ListBuffer(test_set: _ *).asInstanceOf[ListBuffer[Point]])
    getQuerier.sendQueryResponse(
      new QueryResponse(queryId, queryTarget, pj._1.asJava, pj._2, protocol, pj._3, pj._4, pj._5, pj._6)
    )
  }

  /** Pushing the local model to the parameter server(s). */
  def push(): Unit = {
    for ((slice: ParameterDescriptor, index: Int) <- ModelMarshalling(drift = false).zipWithIndex)
      getProxy(index).pushModel(slice).toSync(updateModel)
  }

  /** Pulling the global model from the parameter server(s). */
  def pull(): Unit = {
    for (i <- 0 until getNumberOfHubs)
      getProxy(i).pullModel.toSync(updateModel)
  }

}
