package mlAPI.learners.regression

import ControlAPI.LearnerPOJO
import mlAPI.math.Breeze._
import mlAPI.math.{LabeledPoint, LearningPoint}
import mlAPI.learners.{Learner, PassiveAggressiveLearners}
import mlAPI.parameters.{VectorBias => linearParams}
import mlAPI.scores.Scores

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.collection.JavaConverters._
import breeze.linalg.{DenseVector => BreezeDenseVector}
import mlAPI.utils.Parsing

case class RegressorPA() extends PassiveAggressiveLearners with Regressor with Serializable {

  weights = new linearParams()
  private var epsilon: Double = 0.0

  override def predict(data: LearningPoint): Option[Double] = predictWithMargin(data)

  override def fit(data: LearningPoint): Unit = {
    fitLoss(data)
    ()
  }

  override def fitLoss(data: LearningPoint): Double = {
    predictWithMargin(data) match {
      case Some(prediction) =>
        val label: Double = data.asInstanceOf[LabeledPoint].label
        val loss: Double = Math.abs(label - prediction) - epsilon
        if (loss > 0.0) {
          val Lagrange_Multiplier: Double = LagrangeMultiplier(loss, data)
          val sign: Double = if ((label - prediction) >= 0) 1.0 else -1.0
          weights += linearParams(
            (data.getNumericVector.asBreeze * (Lagrange_Multiplier * sign)).asInstanceOf[BreezeDenseVector[Double]],
            Lagrange_Multiplier * sign)
        }
        loss
      case None =>
        checkParameters(data)
        fitLoss(data)
    }
  }

  override def loss(data: LearningPoint): Double = {
    predictWithMargin(data) match {
      case Some(prediction) => Math.abs(data.asInstanceOf[LabeledPoint].label - prediction) - epsilon
      case None =>
        checkParameters(data)
        loss(data)
    }
  }

  override def loss(batch: ListBuffer[LearningPoint]): Double =
    (for (point <- batch) yield loss(point)).sum / (1.0 * batch.length)

  override def score(testSet: ListBuffer[LearningPoint]): Double =
    Scores.RMSE(testSet.asInstanceOf[ListBuffer[LabeledPoint]], this)

  def setEpsilon(epsilon: Double): RegressorPA = {
    this.epsilon = epsilon
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
              println("Error while trying to update the miniBatchSize hyper parameter of the RegressorPA regressor.")
              e.printStackTrace()
          }
        case "epsilon" =>
          try {
            setEpsilon(value.asInstanceOf[Double])
          } catch {
            case e: Exception =>
              println("Error while trying to update the epsilon hyper parameter of the PA regressor")
              e.printStackTrace()
          }
        case "C" =>
          try {
            setC(value.asInstanceOf[Double])
          } catch {
            case e: Exception =>
              println("Error while trying to update the C hyper parameter of the PA regressor")
              e.printStackTrace()
          }
        case _ =>
      }
    }
    this
  }

  override def toString: String = s"PA regressor ${this.hashCode}"

  override def generatePOJOLearner: LearnerPOJO = {
    new LearnerPOJO("regressorPA",
      Map[String, AnyRef](
        ("C", C.asInstanceOf[AnyRef]),
        ("epsilon", epsilon.asInstanceOf[AnyRef])
      ).asJava,
      Map[String, AnyRef](
        ("a", if(weights == null) null else weights.weights.data.asInstanceOf[AnyRef]),
        ("b", if(weights == null) null else weights.intercept.asInstanceOf[AnyRef])
      ).asJava,
      null
    )
  }

}
