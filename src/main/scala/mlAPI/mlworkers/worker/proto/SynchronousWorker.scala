package mlAPI.mlworkers.worker.proto

import BipartiteTopologyAPI.annotations.{InitOp, ProcessOp, QueryOp}
import ControlAPI.{Prediction, QueryResponse}
import mlAPI.math.{ForecastingPoint, LabeledPoint, UsablePoint, LearningPoint, TrainingPoint, UnlabeledPoint}
import mlAPI.mlworkers.interfaces.Querier
import mlAPI.mlworkers.worker.VectoredWorker
import mlAPI.parameters.VectoredParameters
import mlAPI.parameters.utils.{ParameterDescriptor, WrappedVectoredParameters}
import mlAPI.protocols.IntWrapper
import mlAPI.protocols.periodic.{PushPull, RemoteLearner}

import scala.collection.mutable.ListBuffer
import scala.collection.JavaConverters._

case class SynchronousWorker(override protected var maxMsgParams: Int = 10000)
  extends VectoredWorker[PushPull, Querier] with RemoteLearner {

  println("Synchronous Worker initialized.")

  protocol = "Synchronous Protocol"

  /** Initialization method of the Machine Learning worker node. */
  @InitOp
  def init(): Unit = {
    if (getNodeId != 0)
      pull()
  }

  /** A method for training the SSP worker on a training data point. */
  def train(data: LearningPoint): Unit = {
    fit(data)
    if (processedData >= getMiniBatchSize * miniBatches)
      if (!isWarmedUp && getNodeId == 0) {
        val warmupModel = {
          val wrapped = getMLPipelineParams.get.extractParams(
            getMLPipelineParams.get.asInstanceOf[VectoredParameters],
            false
          ).asInstanceOf[WrappedVectoredParameters]
          ParameterDescriptor(wrapped.getSizes, wrapped.getData, null, null, null, null)
        }
        setWarmed(true)
        setGlobalModelParams(warmupModel)
        for ((hubSubVector, index: Int) <- ModelMarshalling(sendSizes = true, model = getMLPipelineParams.get).zipWithIndex)
          for (slice <- hubSubVector)
            getProxy(index).push(slice)
        processedData = 0
        blockStream()
      } else
        pushPull()
  }

  /** The consumption of a data point by the Machine Learning worker.
   *
   * @param data A data point to be fitted to the model.
   */
  @ProcessOp
  def receiveTuple(data: UsablePoint): Unit = {
    assert(isWarmedUp || (!isWarmedUp && getNodeId == 0))
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

  /** Pushing the local model to the parameter server(s) and waiting for the new global model. */
  def pushPull(): Unit = {
    for ((hubSubVector: Array[ParameterDescriptor], index: Int) <- ModelMarshalling(model = getMLPipelineParams.get).zipWithIndex)
      for (slice <- hubSubVector)
        getProxy(index).pushPull(slice).toSync(updateModel)
    processedData = 0
  }

  /** Requesting the global model from the parameter server(s). */
  def pull(): Unit = {
    setWarmed(false)
    blockStream()
    for (i <- 0 until getNumberOfHubs)
      getProxy(i).pull()
  }

  /**
   * A method called each type the new global model
   * (or a slice of it) arrives from a parameter server.
   */
  override def updateModel(mDesc: ParameterDescriptor): Unit = {
    try {
      val spt: Int = {
        try {
          splits(getCurrentCaller)
        } catch {
          case _: Throwable =>
            assert(mDesc.getMiscellaneous != null, mDesc.getMiscellaneous.head.isInstanceOf[IntWrapper])
            splits.put(getCurrentCaller, mDesc.getMiscellaneous.head.asInstanceOf[IntWrapper].getInt)
            splits(getCurrentCaller)
        }
      }

      if (mDesc.getParams != null)
        if (getNumberOfHubs == 1)
          if (spt == 1)
            if (isWarmedUp)
              updateModels(mDesc)
            else
              warmModel(mDesc)
          else
            updateParameterTree(spt, mDesc, updateModels)
        else
          updateParameterTree(spt, mDesc, updateModels)
      else
        assertWarmup()

    } catch {
      case e: Throwable => throw e
//        e.printStackTrace()
//        throw new RuntimeException("Something went wrong while updating the local model of worker " +
//          getNodeId + " of MLPipeline " + getNetworkID + ".")
    }
  }

  def assertWarmup(): Unit = {
    assert(
      getNodeId == 0 &&
        getMLPipeline.getFittedData == getMiniBatchSize * miniBatches &&
        isBlocked
    )
    unblockStream()
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
    val score = getGlobalPerformance(ListBuffer(testSet: _ *))
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
