package mlAPI.learners.classification

import ControlAPI.LearnerPOJO
import mlAPI.math.Breeze._
import mlAPI.math.{LabeledPoint, LearningPoint}
import mlAPI.learners.Learner
import mlAPI.parameters.{LearningParameters, VectorBias, VectorBiasList}
import breeze.linalg.{DenseVector => BreezeDenseVector}
import mlAPI.parameters.utils.{ParameterDescriptor, SerializableParameters}
import mlAPI.scores.Scores
import mlAPI.utils.Parsing

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.collection.JavaConverters._

/**
  * Multi-class Passive Aggressive Classifier.
  */
case class MultiClassPA() extends Classifier with Serializable {

  override protected var targetLabel: Double = 1.0
  override protected var miniBatchSize: Int = 1
  override protected val parallelizable: Boolean = true
  protected var updateType: String = "PA-II"
  protected var C: Double = 0.01
  protected var weights: VectorBiasList = _
  protected var nClasses: Int = 3

  override def initializeModel(data: LearningPoint): Learner = {
    val vbl: ListBuffer[VectorBias] = ListBuffer[VectorBias]()
    for (_ <- 0 until nClasses) vbl.append(VectorBias(BreezeDenseVector.zeros[Double](data.getNumericVector.size), 0.0))
    weights = VectorBiasList(vbl)
    this
  }

  override def predict(data: LearningPoint): Option[Double] = {
    try {
      var prediction: Int = -1
      var highestScore: Double = -Double.MaxValue
      for ((model: VectorBias, i: Int) <- weights.vectorBiases.zipWithIndex) {
        val currentClassScore = (data.getNumericVector.asBreeze dot model.weights) + model.intercept
        if (currentClassScore > highestScore) {
          prediction = i
          highestScore = currentClassScore
        }
      }
      Some(1.0 * prediction)
    } catch {
      case _: Throwable => None
    }
  }

  override def predict(batch: ListBuffer[LearningPoint]): Array[Option[Double]] = {
    val predictions: ListBuffer[Option[Double]] = ListBuffer[Option[Double]]()
    for (point <- batch)
      predictions append predict(point)
    predictions.toArray
  }

  override def fit(data: LearningPoint): Unit = {
    fitLoss(data)
    ()
  }

  override def fitLoss(data: LearningPoint): Double = {
    predict(data) match {
      case Some(prediction) =>
        val label: Double = data.asInstanceOf[LabeledPoint].label
        val pred: Int = prediction.toInt
        val loss: Double = {
          val exp_weights = weights.vectorBiases(label.toInt)
          val pred_weights = weights.vectorBiases(pred)
          1.0 - (
            ((data.getNumericVector.asBreeze dot exp_weights.weights) + exp_weights.intercept)
              -
              ((data.getNumericVector.asBreeze dot pred_weights.weights) + pred_weights.intercept)
            )
        }
        for ((weight: VectorBias, i: Int) <- weights.vectorBiases.zipWithIndex)
          if (i != label && i != pred)
            ()
          else {
            val t: Double = tau(loss, data)
            if (i == label)
              weight += VectorBias((data.getNumericVector.asBreeze * t).asInstanceOf[BreezeDenseVector[Double]], t)
            else if (i == pred)
              weight -= VectorBias((data.getNumericVector.asBreeze * t).asInstanceOf[BreezeDenseVector[Double]], t)
          }
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
    predict(data) match {
      case Some(prediction) =>
        val label: Double = data.asInstanceOf[LabeledPoint].label
        val pred: Int = prediction.toInt
        val exp_weights = weights.vectorBiases(label.toInt)
        val pred_weights = weights.vectorBiases(pred)
        1.0 - (
          ((data.getNumericVector.asBreeze dot exp_weights.weights) + exp_weights.intercept)
            -
            ((data.getNumericVector.asBreeze dot pred_weights.weights) + pred_weights.intercept)
          )
      case None =>
        checkParameters(data)
        loss(data)
    }
  }

  override def loss(batch: ListBuffer[LearningPoint]): Double =
    (for (point <- batch) yield loss(point)).sum / (1.0 * batch.length)

  override def score(testSet: ListBuffer[LearningPoint]): Double =
    Scores.F1Score(testSet.asInstanceOf[ListBuffer[LabeledPoint]], this)

  private def tau(loss: Double, data: LearningPoint): Double = {
    updateType match {
      case "STANDARD" => loss / (1.0 + 2.0 * ((data.getNumericVector dot data.getNumericVector) + 1.0))
      case "PA-I" => Math.min(C / 2.0, loss / (2.0 * ((data.getNumericVector dot data.getNumericVector) + 1.0)))
      case "PA-II" => 0.5 * (loss /(((data.getNumericVector dot data.getNumericVector) + 1.0) + 1.0 / (2.0 * C)))
    }
  }

  private def setNumberOfClasses(nClasses: Int): Unit = this.nClasses = nClasses

  private def checkParameters(data: LearningPoint): Unit = {
    if (weights == null) {
      initializeModel(data)
    } else {
      if(weights.vectorBiases.head.weights.length != data.getNumericVector.size)
        throw new RuntimeException("Incompatible model and data point size.")
      else
        throw new RuntimeException("Something went wrong while fitting the data point " +
          data + " to learner " + this + ".")
    }
  }

  override def getParameters: Option[LearningParameters] = Option(weights)

  override def setParameters(params: LearningParameters): Learner = {
    assert(params.isInstanceOf[VectorBiasList])
    weights = params.asInstanceOf[VectorBiasList]
    this
  }

  def setC(c: Double): MultiClassPA = {
    this.C = c
    this
  }

  def setType(updateType: String): MultiClassPA = {
    this.updateType = updateType
    this
  }

  override def setParametersFromMap(parameterMap: mutable.Map[String, AnyRef]): Learner = {
    for ((parameter, value) <- parameterMap) {
      parameter match {
        case "weights" =>
          try {
            val vbl: ListBuffer[VectorBias] = ListBuffer[VectorBias]()
            for (v: java.util.List[Double] <- value.asInstanceOf[java.util.List[java.util.List[Double]]].asScala)
              vbl.append(new VectorBias(v.asScala.toArray))
            val new_weights = VectorBiasList(vbl)
            if (weights == null || weights.size == new_weights.size)
              weights = new_weights
            else
              throw new RuntimeException("Invalid size of new weight vector for the multiclass PA classifier.")
          } catch {
            case e: Exception =>
              println("Error while trying to update the weights of the multiclass PA classifier.")
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
              println("Error while trying to update the miniBatchSize hyper parameter of the MultiClassPA classifier.")
              e.printStackTrace()
          }
        case "C" =>
          try {
            setC(value.asInstanceOf[Double])
          } catch {
            case e: Exception =>
              println("Error while trying to update the C hyper parameter of the multiclass PA classifier.")
              e.printStackTrace()
          }
        case "updateType" =>
          try {
            setType(value.asInstanceOf[String])
          } catch {
            case e: Exception =>
              println("Error while trying to update the update type of the multiclass PA classifier.")
              e.printStackTrace()
          }
        case "nClasses" =>
          try {
            setNumberOfClasses(value.asInstanceOf[Double].toInt)
          } catch {
            case e: Exception =>
              println("Error while trying to update the number of classes " +
                "hyper parameter of the multiclass PA classifier.")
              e.printStackTrace()
          }
        case _ =>
      }
    }
    this
  }

  override def toString: String = s"MulticlassPA classifier ${this.hashCode}"

  override def generateParameters: ParameterDescriptor => LearningParameters = {
    if (weights == null)
      new VectorBiasList().generateParameters
    else
      weights.generateParameters
  }

  override def extractParams: (LearningParameters, Boolean) => SerializableParameters = {
    if (weights == null)
      new VectorBiasList().extractParams
    else
      weights.extractParams
  }

  override def extractDivParams: (LearningParameters , Array[_]) => Array[Array[SerializableParameters]] = {
    if (weights == null)
      new VectorBiasList().extractDivParams
    else
      weights.extractDivParams
  }

  override def generatePOJOLearner: LearnerPOJO = {
    new LearnerPOJO("MulticlassPA",
      Map[String, AnyRef](("C", C.asInstanceOf[AnyRef])).asJava,
      Map[String, AnyRef](
        ("weights",
          if(weights == null)
            null
          else
            (for (weight <- weights.vectorBiases) yield weight.flatten.data).toArray.asInstanceOf[AnyRef]
        )
      ).asJava,
      null
    )
  }

}
