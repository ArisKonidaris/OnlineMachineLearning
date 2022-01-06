package mlAPI.pipelines

import ControlAPI.{Request, LearnerPOJO => POJOLearner, PreprocessorPOJO => POJOPreprocessor}
import ControlAPI.{TransformerPOJO => POJOTransformer}
import mlAPI.dataBuffers.DataSet
import mlAPI.math.LearningPoint
import mlAPI.learners.Learner
import mlAPI.learners.classification.nn.NeuralNetwork
import mlAPI.learners.classification.{HoeffdingTreeClassifier, MultiClassPA, PA, SVM}
import mlAPI.learners.clustering.KMeans
import mlAPI.learners.regression.{ORR, RegressorPA}
import mlAPI.parameters.utils.{Bucket, ParameterDescriptor, WithParams, WrappedVectoredParameters}
import mlAPI.parameters.{HTParameters, VectoredParameters}
import mlAPI.preprocessing.{MinMaxScaler, PolynomialFeatures, Preprocessor, RunningMean, StandardScaler}
import mlAPI.protocols.LongWrapper

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.collection.JavaConverters._

case class MLPipeline(private var preprocess: ListBuffer[Preprocessor], private var learner: Learner)
  extends Serializable {

  import MLPipeline._

  def this() = this(ListBuffer[Preprocessor](), null)

  /** The number of data points fitted to the ML pipeline. */
  private var fittedData: Long = 0

  /** The running mean of the loss of the ML pipeline. */
  private var meanLoss: RunningMean = RunningMean()

  /** The cumulative loss of the ML pipeline. */
  private var cumulativeLoss: Double = 0D

  // =================================== Getters ===================================================

  def getPreprocessors: ListBuffer[Preprocessor] = preprocess

  def getLearner: Learner = learner

  def getFittedData: Long = fittedData

  def getMeanLoss: RunningMean = meanLoss

  def getCumulativeLoss: Double = cumulativeLoss

  // =================================== Setters ===================================================

  def setPreprocessors(preprocess: ListBuffer[Preprocessor]): Unit = this.preprocess = preprocess

  def setLearner(learner: Learner): Unit = this.learner = learner

  def setFittedData(fitted_data: Long): Unit = this.fittedData = fitted_data

  def setMeanLoss(meanLoss: RunningMean): Unit = this.meanLoss = meanLoss

  def setCumulativeLoss(cumulative_loss: Double): Unit = this.cumulativeLoss = cumulative_loss

  // =========================== ML Pipeline creation/interaction methods =============================

  def addPreprocessor(preprocessor: Preprocessor): MLPipeline = {
    preprocess = preprocess :+ preprocessor
    this
  }

  def addPreprocessor(preprocessor: Preprocessor, index: Int): MLPipeline = {
    preprocess = (preprocess.slice(0, index) :+ preprocessor) ++ preprocess.slice(index, preprocess.length)
    this
  }

  def removePreprocessor(index: Int): MLPipeline = {
    preprocess = preprocess.slice(0, index) ++ preprocess.slice(index + 1, preprocess.length)
    this
  }

  def addLearner(learner: Learner): MLPipeline = {
    this.learner = learner
    this
  }

  def removeLearner(): MLPipeline = {
    this.learner = null
    this
  }

  def matchPreprocessor(preprocessor: POJOPreprocessor): Option[Preprocessor] = {
    var preProcessor: Option[Preprocessor] = null
    preprocessor.getName match {
      case "PolynomialFeatures" => preProcessor = Some(PolynomialFeatures())
      case "StandardScaler" => preProcessor = Some(StandardScaler())
      case "MinMaxScaler" => preProcessor = Some(MinMaxScaler())
      case _ => None
    }
    preProcessor
  }

  def matchLearner(estimator: POJOLearner): Learner = {
    var learner: Learner = null
    estimator.getName match {
      case "SVM" => learner = new SVM
      case "PA" => learner = new PA
      case "MulticlassPA" => learner = new MultiClassPA
      case "RegressorPA" => learner = new RegressorPA
      case "ORR" => learner = new ORR
      case "NN" => learner = NeuralNetwork()
      case "HT" => learner = HoeffdingTreeClassifier()
      case "K-means" => learner = KMeans()
      case _ => None
    }
    learner
  }

  def configTransformer(transformer: WithParams, preprocessor: POJOTransformer): Unit = {
    val hparams: mutable.Map[String, AnyRef] = preprocessor.getHyperParameters.asScala
    if (hparams != null) transformer.setHyperParametersFromMap(hparams)

    val params: mutable.Map[String, AnyRef] = preprocessor.getParameters.asScala
    if (params != null) transformer.setParametersFromMap(params)

    val structure: mutable.Map[String, AnyRef] = preprocessor.getDataStructure.asScala
    if (structure != null) transformer.setStructureFromMap(structure)
  }

  def createPreProcessor(preprocessor: POJOPreprocessor): Option[Preprocessor] = {
    matchPreprocessor(preprocessor) match {
      case Some(transformer: Preprocessor) =>
        configTransformer(transformer, preprocessor)
        Some(transformer)
      case None => None
    }
  }

  def createLearner(learner: POJOLearner): Learner = {
    val transformer: Learner = matchLearner(learner)
    configTransformer(transformer, learner)
    transformer
  }

  def configureMLPipeline(request: Request): MLPipeline = {
    try {
      val ppContainer: List[POJOPreprocessor] = request.getPreProcessors.asScala.toList
      for (pp: POJOPreprocessor <- ppContainer)
        createPreProcessor(pp) match {
          case Some(preprocessor: Preprocessor) => addPreprocessor(preprocessor)
          case None =>
        }
    } catch {
      case _: java.lang.NullPointerException =>
      case other: Throwable => other.printStackTrace()
    }

    try {
      val lContainer: POJOLearner = request.getLearner
      if (lContainer != null)
        addLearner(createLearner(lContainer))
    } catch {
      case _: java.lang.NullPointerException =>
      case other: Throwable => other.printStackTrace()
    }

    this
  }

  // =================================== ML pipeline basic operations ==============================

  def init(data: LearningPoint): MLPipeline = {
    require(learner != null, "The ML Pipeline must have a learner to fit.")
    pipePoint(data, preprocess, learner.initializeModel)
    this
  }

  def clear(): Unit = {
    fittedData = 0
    preprocess.clear()
    learner = null
  }

  def fit(data: LearningPoint): Unit = {
    require(learner != null, "The ML Pipeline must have a learner to fit data.")
    pipePoint(data, preprocess, learner.fit)
    incrementFitCount()
  }

  def fitLoss(data: LearningPoint): Unit = {
    require(learner != null, "The ML Pipeline must have a learner to fit data.")
    val loss = pipePoint(data, preprocess, learner.fitLoss)
    meanLoss.update(loss)
    incrementCumulativeLoss(loss)
    incrementFitCount()
  }

  def fit(miniBatch: ListBuffer[LearningPoint]): Unit = {
    require(learner != null, "The ML Pipeline must have a learner to fit data.")
    pipePoints(miniBatch, preprocess, learner.fit)
    incrementFitCount(miniBatch.length.asInstanceOf[Long])
  }

  def fitLoss(mini_batch: ListBuffer[LearningPoint]): Unit = {
    require(learner != null, "The ML Pipeline must have a learner to fit data.")
    val loss = pipePoints(mini_batch, preprocess, learner.fitLoss)
    meanLoss.update(loss)
    incrementCumulativeLoss(loss)
    incrementFitCount(mini_batch.length.asInstanceOf[Long])
  }

  def predict(data: LearningPoint): Option[Double] = {
    require(learner != null, "The ML Pipeline must have a learner to make a prediction.")
    pipePoint(data, preprocess, learner.predict)
  }

  def score(testSet: ListBuffer[LearningPoint]): Double = {
    require(learner != null, "Cannot calculate performance. The ML Pipeline doesn't contain a learner.")
    pipePoints(testSet, preprocess, learner.score)
  }

  private def incrementFitCount(miniBatch: Long = 1): Unit = {
    if (fittedData < Long.MaxValue - miniBatch)
      fittedData += miniBatch
    else
      fittedData = Long.MaxValue
  }

  private def incrementCumulativeLoss(loss: Double): Unit = {
    if (cumulativeLoss < Double.MaxValue - loss)
      cumulativeLoss += loss
    else
      cumulativeLoss = Double.MaxValue
  }

  def merge(mlPipeline: MLPipeline): MLPipeline = {
    incrementFitCount(mlPipeline.getFittedData)
    preprocess = mlPipeline.getPreprocessors
    learner = mlPipeline.getLearner
    this
  }

  def generateDescriptor(): ParameterDescriptor = {
    if (learner != null && learner.getParameters.isDefined) {
      getLearner.getParameters.get match {
        case _: VectoredParameters =>
          val wrapped = getLearner.extractParams(getLearner.getParameters.get, false)
            .asInstanceOf[WrappedVectoredParameters]
          ParameterDescriptor(
            wrapped.getSizes,
            wrapped.getData,
            Bucket(0, getLearner.getParameters.get.getSize - 1),
            null,
            null,
            LongWrapper(fittedData)
          )
        case _: HTParameters =>
          ParameterDescriptor(
            null,
            null,
            null,
            getLearner.extractParams(getLearner.getParameters.get, false),
            null,
            LongWrapper(fittedData)
          )
      }
    } else
      new ParameterDescriptor()
  }

  def generatePOJO: (List[POJOPreprocessor], POJOLearner, Long, Double, Double) = {
    val prPJ = (for (preprocessor <- getPreprocessors) yield preprocessor.generatePOJOPreprocessor).toList
    val lrPJ = getLearner.generatePOJOLearner
    (prPJ, lrPJ, fittedData, meanLoss.getMean, cumulativeLoss)
  }

  def generatePOJO(testSet: ListBuffer[LearningPoint]): (List[POJOPreprocessor], POJOLearner, Long, Double, Double, Double) = {
    val genPJ = generatePOJO
    (genPJ._1, genPJ._2, genPJ._3, genPJ._4, genPJ._5, score(testSet))
  }

}

object MLPipeline {

  // =================================== Factory methods ===========================================

  def apply(): MLPipeline = new MLPipeline()

  // ====================================== Operations =============================================

  @scala.annotation.tailrec
  final def pipePoint[T](data: LearningPoint, list: ListBuffer[Preprocessor], f: LearningPoint => T): T = {
    if (list.isEmpty) f(data) else pipePoint(list.head.transform(data), list.tail, f)
  }

  @scala.annotation.tailrec
  final def pipePoints[T](data: ListBuffer[LearningPoint], list: ListBuffer[Preprocessor], f: ListBuffer[LearningPoint] => T): T = {
    if (list.isEmpty) f(data) else pipePoints(list.head.transform(data), list.tail, f)
  }

}
