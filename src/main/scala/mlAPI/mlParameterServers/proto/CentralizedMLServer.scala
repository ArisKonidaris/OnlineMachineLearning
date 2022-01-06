package mlAPI.mlParameterServers.proto

import BipartiteTopologyAPI.NodeInstance
import BipartiteTopologyAPI.annotations.{InitOp, MergeOp, ProcessOp, QueryOp}
import ControlAPI.{Prediction, QueryResponse, Request}
import mlAPI.dataBuffers.DataSet
import mlAPI.math.{ForecastingPoint, LabeledPoint, LearningPoint, TrainingPoint, UnlabeledPoint, UsablePoint}
import mlAPI.mlworkers.interfaces.Querier
import mlAPI.parameters.utils.ParameterDescriptor
import mlAPI.pipelines.MLPipeline
import mlAPI.preprocessing.RunningMean
import mlAPI.protocols.statistics.SingleLearnerStatistics
import mlAPI.protocols.{CentralizedLearner, IntWrapper, LongWrapper, RemoteForwarder}

import java.io.Serializable
import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ListBuffer

case class CentralizedMLServer() extends NodeInstance[RemoteForwarder, Querier] with CentralizedLearner {

  protected val statistics: SingleLearnerStatistics = SingleLearnerStatistics()

  protected val protocol: String = "CentralizedLearner"

  protected var count: Int = 0

  protected var model: MLPipeline = new MLPipeline()

  protected var testSet: DataSet[TrainingPoint] = new DataSet[TrainingPoint](256)

  protected var cumulativeLoss: Double = 0.0

  protected var miniBatches: Int = 256

  protected var cnt: Int = 0

  /** An index counter. */
  protected var idx: Int = 0

  /** The learning curve for the ML pipeline. */
  protected var learningCurve: DataSet[(Double, Long)] = new DataSet[(Double, Long)](100000)

  @InitOp
  def init(): Unit = ()

  @ProcessOp
  def receiveTuple[T <: Serializable](data: T): Unit = ()

  @MergeOp
  def merge(pS: Array[CentralizedMLServer]): CentralizedMLServer = this

  @QueryOp
  def query(qId: Long, qT: Int, pr: Array[java.io.Serializable]): Unit = ()

  override def forward(record: UsablePoint): Unit = {
    record match {
      case tP: TrainingPoint =>
        if (count >= 8) {
          testSet.append(tP) match {
            case Some(point: TrainingPoint) =>
              model.fitLoss(point.trainingPoint)
              cnt += 1
            case None =>
          }
        } else {
          model.fitLoss(tP.trainingPoint)
          cnt += 1
        }
        count += 1
        if (count == 10)
          count = 0
        if (cnt == miniBatches) {
          learningCurve.append(((model.getCumulativeLoss - cumulativeLoss) / miniBatches, model.getFittedData))
          cumulativeLoss = model.getCumulativeLoss
          cnt = 0
        }
      case fP: ForecastingPoint =>
        val prediction: Double = {
          try {
            model.predict(fP.forecastingPoint) match {
              case Some(prediction: Double) => prediction
              case None => Double.NaN
            }
          } catch {
            case _: Throwable => Double.NaN
          }
        }
        getQuerier.sendQueryResponse(new Prediction(getNetworkID(), fP.toDataInstance, prediction))
    }
  }

  override def describe(qDesc: ParameterDescriptor): Unit = {
    val queryId: Long = qDesc.getMiscellaneous(0).asInstanceOf[LongWrapper].getLong
    val queryTarget: Int = qDesc.getMiscellaneous(1).asInstanceOf[IntWrapper].getInt
    val pj = model.generatePOJO
    val tS: ListBuffer[LearningPoint] = testSet.dataBuffer.map {
      case TrainingPoint(trainingPoint) => trainingPoint
    }
    val score = model.score(tS)
    if (queryId == -1)
      getQuerier.sendQueryResponse(
        new QueryResponse(-1,
          queryTarget,
          null,
          null,
          null,
          model.getFittedData,
          null,
          pj._5,
          score)
      )
    else
      getQuerier.sendQueryResponse(
        new QueryResponse(queryId, queryTarget, pj._1.asJava, pj._2, protocol, pj._3, pj._4, pj._5, score)
      )
  }

  def configureParameterServer(request: Request): CentralizedMLServer = {

    // Setting the ML node parameters.
    val config: mutable.Map[String, AnyRef] = request.getTrainingConfiguration.asScala

    if (config == null)
      throw new RuntimeException("Empty training configuration map.")

    // Setting the ML pipeline and the global model.
    model.configureMLPipeline(request)

    this
  }

  def fitted: Long = model.getFittedData

  def getProtocolStatistics: SingleLearnerStatistics = statistics

  def getLearningCurve: DataSet[(Double,Long)] = learningCurve

  def getIdx: Int = idx

  def setLearningCurve(learningCurve: DataSet[(Double,Long)]): Unit = this.learningCurve = learningCurve

  def setIdx(idx: Int): Unit = this.idx = idx

}
