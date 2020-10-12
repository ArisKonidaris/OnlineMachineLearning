package mlAPI.learners.classification

import mlAPI.learners.{Parallelizable, PassiveAggressiveLearners}
import mlAPI.math.Breeze._
import mlAPI.math.{LabeledPoint, Point}
import mlAPI.parameters.VectorBias
import mlAPI.scores.Scores

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.collection.JavaConverters._

/**
  * Passive Aggressive Classifier.
  */
case class PA() extends PassiveAggressiveLearners with Classifier with Parallelizable with Serializable {

  override def predict(data: Point): Option[Double] = {
    predictWithMargin(data) match {
      case Some(pred) => if (pred >= 0.0) Some(1.0) else Some(-1.0)
      case None => Some(Double.MinValue)
    }
  }

  override def fit(data: Point): Unit = {
    fitLoss(data)
    ()
  }

  override def fitLoss(data: Point): Double = {
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

  override def score(test_set: ListBuffer[Point]): Double =
    Scores.F1Score(test_set.asInstanceOf[ListBuffer[LabeledPoint]], this)

  private def createLabel(label: Double): Double = if (label == 0.0) -1.0 else label

  override def toString: String = s"PA classifier ${this.hashCode}"

  override def setHyperParametersFromMap(hyperParameterMap: mutable.Map[String, AnyRef]): mlAPI.learners.Learner = {
    for ((hyperparameter, value) <- hyperParameterMap) {
      hyperparameter match {
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

  override def generatePOJOLearner: ControlAPI.Learner = {
    new ControlAPI.Learner("PA",
      Map[String, AnyRef](("C", C.asInstanceOf[AnyRef])).asJava,
      Map[String, AnyRef](
        ("a", if(weights == null) null else weights.weights.data.asInstanceOf[AnyRef]),
        ("b", if(weights == null) null else weights.intercept.asInstanceOf[AnyRef])
      ).asJava
    )
  }

}
