package mlAPI.learners.classification

import ControlAPI.LearnerPOJO
import mlAPI.math.Breeze._
import mlAPI.learners.Learner
import mlAPI.math.{LabeledPoint, LearningPoint}
import mlAPI.parameters.{LearningParameters, VectorBias}
import mlAPI.scores.Scores
import breeze.linalg.{DenseVector => BreezeDenseVector}
import mlAPI.parameters.utils.{ParameterDescriptor, SerializableParameters}
import mlAPI.utils.Parsing

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ListBuffer

/**
 * Support Vector Machine classifier.
 */
case class SVM() extends Classifier with Serializable {

  override protected var targetLabel: Double = 1.0
  override protected var miniBatchSize: Int = 1
  override protected val parallelizable: Boolean = true
  protected var C: Double = 0.01
  protected var weights: VectorBias = _
  protected var count: Long = 0L

  override def initializeModel(data: LearningPoint): Learner = {
    weights = VectorBias(BreezeDenseVector.zeros[Double](data.getNumericVector.size), 0.0)
    this
  }

  def predictWithMargin(data: LearningPoint): Option[Double] = {
    try {
      Some((data.getNumericVector.asBreeze dot weights.weights) + weights.intercept)
    } catch {
      case _: Throwable => None
    }
  }

  override def predict(data: LearningPoint): Option[Double] = {
    predictWithMargin(data) match {
      case Some(pred) => if (pred >= 0.0) Some(1.0) else Some(-1.0)
      case None => Some(Double.MinValue)
    }
  }

  override def predict(batch: ListBuffer[LearningPoint]): Array[Option[Double]] = {
    val predictions: ListBuffer[Option[Double]] = ListBuffer[Option[Double]]()
    for (point <- batch)
      predictions append predict(point)
    predictions.toArray
  }

  override def fit(data: LearningPoint): Unit = {
    if (count < Long.MaxValue) fitLoss(data)
    ()
  }

  override def fitLoss(data: LearningPoint): Double = {
    if (count == Long.MaxValue)
      0
    else
      predictWithMargin(data) match {
        case Some(prediction) =>
          val label: Double = createLabel(data.asInstanceOf[LabeledPoint].label)
          val sign: Double = if (label * prediction < 1.0) 1.0 else 0.0
          val loss: Double = Math.max(0.0, 1.0 - label * prediction)

          val direction = VectorBias(weights.weights - C * label * sign * data.getNumericVector.asBreeze, -label * sign)

          count += 1
          weights -= (direction / count)

          loss
        case None =>
          checkParameters(data)
          fitLoss(data)
      }
  }

  override def fit(batch: ListBuffer[LearningPoint]): Unit = {
    fitLoss(batch)
    ()
  }

  override def fitLoss(batch: ListBuffer[LearningPoint]): Double = (for (point <- batch) yield fitLoss(point)).sum

  override def loss(data: LearningPoint): Double = {
    predictWithMargin(data) match {
      case Some(prediction) => Math.max(0.0, 1.0 - createLabel(data.asInstanceOf[LabeledPoint].label) * prediction)
      case None =>
        checkParameters(data)
        loss(data)
    }
  }

  override def loss(batch: ListBuffer[LearningPoint]): Double =
    (for (point <- batch) yield loss(point)).sum / (1.0 * batch.length)

  override def score(testSet: ListBuffer[LearningPoint]): Double =
    Scores.F1Score(testSet.asInstanceOf[ListBuffer[LabeledPoint]], this)

  private def createLabel(label: Double): Double = if (label == 0.0) -1.0 else label

  private def checkParameters(data: LearningPoint): Unit = {
    if (weights == null) {
      initializeModel(data)
    } else {
      if (weights.weights.size != data.getNumericVector.size)
        throw new RuntimeException("Incompatible model and data point size.")
      else
        throw new RuntimeException("Something went wrong while fitting the data point " +
          data + " to learner " + this + ".")
    }
  }

  override def getParameters: Option[LearningParameters] = Some(weights)

  override def setParameters(params: LearningParameters): Learner = {
    assert(params.isInstanceOf[VectorBias])
    weights = params.asInstanceOf[VectorBias]
    this
  }

  def setC(c: Double): Unit = this.C = c

  def setCount(count: Long): Unit = this.count = count

  override def setParametersFromMap(parameterMap: mutable.Map[String, AnyRef]): Learner = {
    for ((parameter, value) <- parameterMap) {
      parameter match {
        case "a" =>
          try {
            val new_weights = BreezeDenseVector[Double](value.asInstanceOf[java.util.List[Double]].asScala.toArray)
            if (weights == null || weights.weights.size == new_weights.size)
              weights.weights = new_weights
            else
              throw new RuntimeException("Invalid size of new weight vector for the SVM classifier.")
          } catch {
            case e: Exception =>
              println("Error while trying to update the weights of the SVM classifier.")
              e.printStackTrace()
          }
        case "b" =>
          try {
            weights.intercept = value.asInstanceOf[Double]
          } catch {
            case e: Exception =>
              println("Error while trying to update the intercept of the SVM classifier.")
              e.printStackTrace()
          }
        case _ =>
      }
    }
    this
  }

  override def setHyperParametersFromMap(hyperParameterMap: mutable.Map[String, AnyRef]): Learner = {
    for ((hyperparameter, value) <- hyperParameterMap) {
      hyperparameter match {
        case "miniBatchSize" =>
          try {
            miniBatchSize = Parsing.IntegerParsing(hyperParameterMap, "miniBatchSize", 1)
          } catch {
            case e: Exception =>
              println("Error while trying to update the miniBatchSize hyper parameter of the SVM classifier.")
              e.printStackTrace()
          }
        case "C" =>
          try {
            setC(value.asInstanceOf[Double])
          } catch {
            case e: Exception =>
              println("Error while trying to update the C hyper parameter of the SVM classifier.")
              e.printStackTrace()
          }
        case "count" =>
          try {
            setCount(value.asInstanceOf[Double].toLong)
          } catch {
            case e: Exception =>
              println("Error while trying to update the count hyper parameter of the SVM classifier.")
              e.printStackTrace()
          }
        case _ =>
      }
    }
    this
  }

  override def toString: String = s"SVM classifier ${this.hashCode}"

  override def generateParameters: ParameterDescriptor => LearningParameters = {
    if (weights == null)
      new VectorBias().generateParameters
    else
      weights.generateParameters
  }

  override def extractParams: (LearningParameters, Boolean) => SerializableParameters = {
    if (weights == null)
      new VectorBias().extractParams
    else
      weights.extractParams
  }

  override def extractDivParams: (LearningParameters, Array[_]) => Array[Array[SerializableParameters]] = {
    if (weights == null)
      new VectorBias().extractDivParams
    else
      weights.extractDivParams
  }

  override def generatePOJOLearner: LearnerPOJO = {
    new LearnerPOJO("SVM",
      Map[String, AnyRef](("C", C.asInstanceOf[AnyRef])).asJava,
      Map[String, AnyRef](
        ("a", if (weights == null) null else weights.weights.data.asInstanceOf[AnyRef]),
        ("b", if (weights == null) null else weights.intercept.asInstanceOf[AnyRef])
      ).asJava,
      null
    )
  }

}
