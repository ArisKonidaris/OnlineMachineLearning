package mlAPI.mlworkers.worker.proto

import BipartiteTopologyAPI.annotations.{InitOp, ProcessOp, QueryOp}
import ControlAPI.{Prediction, QueryResponse}
import mlAPI.math.{ForecastingPoint, LabeledPoint, UsablePoint, LearningPoint, TrainingPoint, UnlabeledPoint}
import mlAPI.mlworkers.interfaces.Querier
import mlAPI.mlworkers.worker.VectoredWorker
import mlAPI.parameters.utils.ParameterDescriptor
import mlAPI.protocols.periodic.{Push, RemoteLearner}

import scala.collection.mutable.ListBuffer
import scala.collection.JavaConverters._

case class SingleWorker(override protected var maxMsgParams: Int = 10000)
  extends VectoredWorker[Push, Querier] with RemoteLearner {

  println("Single Worker initialized.")

  protocol = "CentralizedTraining"

  /** Initialization method of the Machine Learning worker node. */
  @InitOp
  def init(): Unit = ()

  /** A method for training the single worker on a training data point. */
  def train(data: LearningPoint): Unit = {
    fit(data)
    if (processedData >= getMiniBatchSize * miniBatches)
      push()
  }

  /** The consumption of a data point by the Machine Learning worker.
   *
   * @param data A data point to be fitted to the model.
   */
  @ProcessOp
  def receiveTuple(data: UsablePoint): Unit = {
    if (mlPipeline.getFittedData % 10000 == 0)
      println(mlPipeline.getFittedData)
    data match {
      case TrainingPoint(trainingPoint) => train(trainingPoint)
      case ForecastingPoint(forecastingPoint) =>
        val prediction: Double = {
          try {
            globalModel.predict(forecastingPoint) match {
              case Some(prediction: Double) => prediction
              case None => Double.NaN
            }
          } catch {
            case _: Throwable => Double.NaN
          }
        }
        getQuerier.sendQueryResponse(new Prediction(getNetworkID(), forecastingPoint.toDataInstance, prediction))
    }
  }

  /** Pushing the local model to the parameter server(s). */
  def push(): Unit = {
    for ((hubSubVector, index: Int) <- ModelMarshalling(model = getMLPipelineParams.get, sendSizes = true).zipWithIndex)
      for (slice <- hubSubVector)
        getProxy(index).push(slice)
    processedData = 0
  }

  /** This method responds to a query for the Machine Learning worker.
   *
   * @param predicates The predicated of the query.
   */
  @QueryOp
  def query(queryId: Long, queryTarget: Int, predicates: (Double, Array[UsablePoint])): Unit = {
    val pj = mlPipeline.generatePOJO
    val testSet: Array[LearningPoint] = predicates._2.map {
      case TrainingPoint(trainingPoint) => trainingPoint
      case ForecastingPoint(forecastingPoint) => forecastingPoint
      case labeledPoint: LabeledPoint => labeledPoint
      case unlabeledPoint: UnlabeledPoint => unlabeledPoint
    }
    val score = mlPipeline.score(ListBuffer(testSet: _ *))
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

  override def updateModel(model: ParameterDescriptor): Unit = ()

}
