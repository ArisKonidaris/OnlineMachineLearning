package mlAPI.mlworkers.worker.proto

import BipartiteTopologyAPI.annotations.{InitOp, ProcessOp, QueryOp}
import ControlAPI.{Prediction, QueryResponse, Request}
import mlAPI.learners.SGDUpdate
import mlAPI.math.{ForecastingPoint, LabeledPoint, LearningPoint, TrainingPoint, UnlabeledPoint, UsablePoint}
import mlAPI.mlworkers.interfaces.Querier
import mlAPI.mlworkers.worker.{MLWorker, VectoredWorker}
import mlAPI.parameters.VectoredParameters
import mlAPI.parameters.utils.{ParameterDescriptor, WrappedVectoredParameters}
import mlAPI.protocols.{DoubleWrapper, IntWrapper}
import mlAPI.protocols.periodic.{PullPush, RemoteLearner}
import mlAPI.utils.Parsing

import scala.collection.mutable
import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer

case class EASGDWorker(override protected var maxMsgParams: Int = 10000)
  extends VectoredWorker[PullPush, Querier] with RemoteLearner {

  println("Elastic Averaging Worker initialized.")

  protocol = "Elastic Averaging SGD protocol"

  /** The learning rate. */
  protected var n: Double = 0.001

  /** The exploration term. */
  protected var rho: Double = _

  /** The moving rate hyper-parameter. */
  protected var alpha: Double = _

  /** Initialization method of the Machine Learning worker node. */
  @InitOp
  def init(): Unit = {
    rho = (math.pow(0.9, 8) / getNumberOfSpokes) / 0.001
    alpha = math.pow(0.9, 8) / getNumberOfSpokes
    if (getNodeId != 0)
      pull()
  }

  /** Requesting the global model from the parameter server(s). */
  def pull(): Unit = {
    blockStream()
    for (i <- 0 until getNumberOfHubs)
      getProxy(i).pull()
  }

  /** A method for training the EASGD worker on a training data point. */
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
        for ((hubSubVector, index: Int) <- sendLoss(ModelMarshalling(sendSizes = true, model = getMLPipelineParams.get)).zipWithIndex)
          for (slice <- hubSubVector)
            getProxy(index).push(slice)
        processedData = 0
        blockStream()
      } else
        pull()
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
              updateCenterVariable(mDesc)
            else
              warmModel(mDesc)
          else
            updateParameterTree(spt, mDesc, updateCenterVariable)
        else
          updateParameterTree(spt, mDesc, updateCenterVariable)
      else
        assertWarmup()

    } catch {
      case e: Throwable =>
        e.printStackTrace()
        throw new RuntimeException("Something went wrong while updating the center model of worker " +
          getNodeId + " of MLPipeline " + getNetworkID + ".")
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

  def updateCenterVariable(mDesc: ParameterDescriptor): Unit = {
    if (mDesc.getParamSizes == null)
      mDesc.setParamSizes(getMLPipeline.getLearner.getParameters.get.sizes)
    if (mDesc.getFitted != null)
      mlPipeline.setFittedData(mDesc.getFitted.getLong)
    setGlobalModelParams(mDesc)
    val dif = getMLPipelineParams.get.asInstanceOf[VectoredParameters] - getGlobalParams.get.asInstanceOf[VectoredParameters]
    getMLPipelineParams.get.asInstanceOf[VectoredParameters] -=
      (dif.asInstanceOf[VectoredParameters] * alpha).asInstanceOf[VectoredParameters]
    for ((hubSubVector, index: Int) <- sendLoss(ModelMarshalling(model = dif)).zipWithIndex)
      for (slice <- hubSubVector)
        getProxy(index).push(slice)
    processedData = 0
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
          score
        )
      )
    else {
      if (getNodeId == 0)
        getQuerier.sendQueryResponse(
          new QueryResponse(queryId, queryTarget, pj._1.asJava, pj._2, protocol, pj._3, pj._4, pj._5, score)
        )
      else {
        getQuerier.sendQueryResponse(
          new QueryResponse(queryId, queryTarget, null, null, null, processedData, pj._4, pj._5, score)
        )
      }
    }
  }

  def setLearningRate(lr: Double): Unit = {
    n = lr
    if (getMLPipeline.getLearner != null) {
      require(getMLPipeline.getLearner.isInstanceOf[SGDUpdate])
      getMLPipeline.getLearner.asInstanceOf[SGDUpdate].setLearningRate(lr)
    }
    alpha = n * rho
  }

  def setRho(r: Double): Unit = {
    rho = r
    alpha = n * rho
  }

  override def configureWorker(request: Request): MLWorker[PullPush, Querier] = {
    super.configureWorker(request)

    require(getMLPipeline.getLearner.isInstanceOf[SGDUpdate])

    // Setting the ML node parameters.
    val config: mutable.Map[String, AnyRef] = request.getTrainingConfiguration.asScala

    if (config.contains("learningRate"))
      setLearningRate(Parsing.DoubleParsing(config, "learningRate", 0.001))

    if (config.contains("rho"))
      setRho(Parsing.DoubleParsing(config, "rho", (math.pow(0.9, 8) / getNumberOfSpokes) / 0.001))

    this
  }

}
