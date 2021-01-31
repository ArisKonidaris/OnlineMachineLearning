package mlAPI.learners.classification

import ControlAPI.LearnerPOJO
import mlAPI.learners.PassiveAggressiveLearners
import mlAPI.math.Breeze._
import mlAPI.math.{LabeledPoint, LearningPoint}
import mlAPI.parameters.VectorBias
import mlAPI.scores.Scores
import mlAPI.utils.Parsing

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.collection.JavaConverters._

/**
  * Passive Aggressive Classifier.
  */
case class PA() extends PassiveAggressiveLearners with Classifier with Serializable {

  override protected var targetLabel: Double = 1.0

  override def predict(data: LearningPoint): Option[Double] = {
    predictWithMargin(data) match {
      case Some(pred) => if (pred >= 0.0) Some(1.0) else Some(-1.0)
      case None => Some(Double.MinValue)
    }
  }

  override def fit(data: LearningPoint): Unit = {
    fitLoss(data)
    ()
  }

  override def fitLoss(data: LearningPoint): Double = {
    predictWithMargin(data) match {
      case Some(prediction) =>
        val label: Double = createLabel(data.asInstanceOf[LabeledPoint].label)
        val loss: Double = Math.max(0.0, 1.0 - label * prediction)
        if (loss > 0.0) {
          val lagrangeMultiplier: Double = LagrangeMultiplier(loss, data)
          weights +=
            VectorBias(data.getNumericVector.asDenseBreeze * (lagrangeMultiplier * label), lagrangeMultiplier * label)
        }
        loss
      case None =>
        checkParameters(data)
        fitLoss(data)
    }
  }

  override def score(testSet: ListBuffer[LearningPoint]): Double =
    Scores.F1Score(testSet.asInstanceOf[ListBuffer[LabeledPoint]], this)

  private def createLabel(label: Double): Double = if (label != targetLabel) -1.0 else label

  override def toString: String = s"PA classifier ${this.hashCode}"

  override def setHyperParametersFromMap(hyperParameterMap: mutable.Map[String, AnyRef]): mlAPI.learners.Learner = {
    for ((hyperparameter, value) <- hyperParameterMap) {
      hyperparameter match {
        case "miniBatchSize" =>
          try {
            miniBatchSize = Parsing.IntegerParsing(hyperParameterMap, "miniBatchSize", 1)
          } catch {
            case e: Exception =>
              println("Error while trying to update the miniBatchSize hyper parameter of the PA classifier.")
              e.printStackTrace()
          }
        case "C" =>
          try {
            setC(value.asInstanceOf[Double])
          } catch {
            case e: Exception =>
              println("Error while trying to update the C hyper parameter of PA classifier.")
              e.printStackTrace()
          }
        case "updateType" =>
          try {
            setType(value.asInstanceOf[String])
          } catch {
            case e: Exception =>
              println("Error while trying to update the update type of PA classifier.")
              e.printStackTrace()
          }
        case _ =>
      }
    }
    this
  }

  override def generatePOJOLearner: LearnerPOJO = {
    new LearnerPOJO("PA",
      Map[String, AnyRef](("C", C.asInstanceOf[AnyRef])).asJava,
      Map[String, AnyRef](
        ("a", if(weights == null) null else weights.weights.data.asInstanceOf[AnyRef]),
        ("b", if(weights == null) null else weights.intercept.asInstanceOf[AnyRef])
      ).asJava,
      null
    )
  }

}
