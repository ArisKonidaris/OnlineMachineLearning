package mlAPI.pipelines

import ControlAPI.{Request, LearnerPOJO => POJOLearner, PreprocessorPOJO => POJOPreprocessor}
import ControlAPI.{TransformerPOJO => POJOTransformer}
import mlAPI.dataBuffers.DataSet
import mlAPI.math.{Point, Vector}
import mlAPI.learners.Learner
import mlAPI.learners.classification.{MultiClassPA, PA, SVM}
import mlAPI.learners.regression.{ORR, regressorPA}
import mlAPI.parameters.{Bucket, VectoredParameters, HTParameters, ParameterDescriptor, WithParams}
import mlAPI.preprocessing.{PolynomialFeatures, Preprocessor, StandardScaler}

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.collection.JavaConverters._

case class MLPipeline(private var preprocess: ListBuffer[Preprocessor], private var learner: Learner)
  extends Serializable {

  import MLPipeline._

  def this() = this(ListBuffer[Preprocessor](), null)

  /** The number of data points fitted to the ML pipeline. */
  private var fitted_data: Long = 0

  /** A buffer containing the last 500 losses of the pipeline. */
  private var losses: DataSet[Double] = new DataSet[Double](100)

  /** The cumulative loss of the pipeline. */
  private var cumulative_loss: Double = 0D

  // =================================== Getters ===================================================

  def getPreprocessors: ListBuffer[Preprocessor] = preprocess

  def getLearner: Learner = learner

  def getFittedData: Long = fitted_data

  def getLosses: DataSet[Double] = losses

  def getCumulativeLoss: Double = cumulative_loss

  // =================================== Setters ===================================================

  def setPreprocessors(preprocess: ListBuffer[Preprocessor]): Unit = this.preprocess = preprocess

  def setLearner(learner: Learner): Unit = this.learner = learner

  def setFittedData(fitted_data: Long): Unit = this.fitted_data = fitted_data

  def setLosses(losses: DataSet[Double]): Unit = this.losses = losses

  def setCumulativeLoss(cumulative_loss: Double): Unit = this.cumulative_loss = cumulative_loss

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
      case "regressorPA" => learner = new regressorPA
      case "ORR" => learner = new ORR
      case _ => None
    }
    learner
  }

  def configTransformer(transformer: WithParams, preprocessor: POJOTransformer): Unit = {
    val hparams: mutable.Map[String, AnyRef] = preprocessor.getHyperparameters.asScala
    if (hparams != null) transformer.setHyperParametersFromMap(hparams)

    val params: mutable.Map[String, AnyRef] = preprocessor.getParameters.asScala
    if (params != null) transformer.setParametersFromMap(params)
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
      val ppContainer: List[POJOPreprocessor] = request.getPreprocessors.asScala.toList
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
      if (lContainer != null) addLearner(createLearner(lContainer))
    } catch {
      case _: java.lang.NullPointerException =>
      case other: Throwable => other.printStackTrace()
    }

    this
  }

  // =================================== ML pipeline basic operations ==============================

  def init(data: Point): MLPipeline = {
    require(learner != null, "The ML Pipeline must have a learner to fit.")
    pipePoint(data, preprocess, learner.initialize_model)
    this
  }

  def clear(): Unit = {
    fitted_data = 0
    preprocess.clear()
    learner = null
  }

  def fit(data: Point): Unit = {
    require(learner != null, "The ML Pipeline must have a learner to fit data.")
    pipePoint(data, preprocess, learner.fit)
    incrementFitCount()
  }

  def fitLoss(data: Point): Unit = {
    require(learner != null, "The ML Pipeline must have a learner to fit data.")
    val loss = pipePoint(data, preprocess, learner.fitLoss)
    losses.append(loss)
    cumulative_loss += loss
    incrementFitCount()
  }

  def fit(mini_batch: ListBuffer[Point]): Unit = {
    require(learner != null, "The ML Pipeline must have a learner to fit data.")
    pipePoints(mini_batch, preprocess, learner.fit)
    incrementFitCount(mini_batch.length.asInstanceOf[Long])
  }

  def fitLoss(mini_batch: ListBuffer[Point]): Unit = {
    require(learner != null, "The ML Pipeline must have a learner to fit data.")
    val loss = pipePoints(mini_batch, preprocess, learner.fitLoss)
    losses.append(loss)
    cumulative_loss += loss
    incrementFitCount(mini_batch.length.asInstanceOf[Long])
  }

  def predict(data: Point): Option[Double] = {
    require(learner != null, "The ML Pipeline must have a learner to make a prediction.")
    pipePoint(data, preprocess, learner.predict)
  }

  def score(testSet: ListBuffer[Point]): Double = {
    require(learner != null, "Cannot calculate performance. The ML Pipeline doesn't contain a learner.")
    pipePoints(testSet, preprocess, learner.score)
  }

  private def incrementFitCount(mini_batch: Long = 1): Unit = {
    if (fitted_data < Long.MaxValue - mini_batch) fitted_data += mini_batch else fitted_data = Long.MaxValue
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
          val (sizes, parameters) = {
            getLearner
              .getSerializedParams(
                getLearner.getParameters.get,
                Array(false, Bucket(0, getLearner.getParameters.get.getSize - 1))
              ).asInstanceOf[(Array[Int], Vector)]
          }
          ParameterDescriptor(
            sizes,
            parameters,
            Bucket(0, getLearner.getParameters.get.getSize - 1),
            null,
            null,
            fitted_data
          )
        case _: HTParameters =>
          ParameterDescriptor(
            null,
            null,
            null,
            getLearner.getSerializedParams(getLearner.getParameters.get, null),
            null,
            fitted_data
          )
      }
    } else new ParameterDescriptor()
  }

  def generatePOJO: (List[POJOPreprocessor], POJOLearner, Long, Double, Double) = {
    val prPJ = (for (preprocessor <- getPreprocessors) yield preprocessor.generatePOJOPreprocessor).toList
    val lrPJ = getLearner.generatePOJOLearner
    (prPJ, lrPJ, fitted_data, losses.data_buffer.sum, cumulative_loss)
  }

  def generatePOJO(testSet: ListBuffer[Point]): (List[POJOPreprocessor], POJOLearner, Long, Double, Double, Double) = {
    val genPJ = generatePOJO
    (genPJ._1, genPJ._2, genPJ._3, genPJ._4, genPJ._5, score(testSet))
  }

}

object MLPipeline {

  // =================================== Factory methods ===========================================

  def apply(): MLPipeline = new MLPipeline()

  // ====================================== Operations =============================================

  @scala.annotation.tailrec
  final def pipePoint[T](data: Point, list: ListBuffer[Preprocessor], f: Point => T): T = {
    if (list.isEmpty) f(data) else pipePoint(list.head.transform(data), list.tail, f)
  }

  @scala.annotation.tailrec
  final def pipePoints[T](data: ListBuffer[Point], list: ListBuffer[Preprocessor], f: ListBuffer[Point] => T): T = {
    if (list.isEmpty) f(data) else pipePoints(list.head.transform(data), list.tail, f)
  }

}