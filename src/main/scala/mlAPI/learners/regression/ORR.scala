package mlAPI.learners.regression

import ControlAPI.LearnerPOJO
import mlAPI.math.Breeze._
import mlAPI.math.{LabeledPoint, Point}
import mlAPI.learners.{Learner, OnlineLearner}
import mlAPI.parameters.{LearningParameters, MatrixBias, ParameterDescriptor}

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.collection.JavaConverters._
import breeze.linalg.{DenseVector => BreezeDenseVector, _}
import mlAPI.scores.Scores

/**
  * Online Ridge Regression.
  */
case class ORR() extends OnlineLearner with Regressor with Serializable {

  override protected val parallelizable: Boolean = true

  protected var weights: MatrixBias = _

  protected var lambda: Double = 0.0

  override def initialize_model(data: Point): Unit = weights = model_init(data.getNumericVector.size + 1)

  override def predict(data: Point): Option[Double] = {
    try {
      val x: BreezeDenseVector[Double] = add_bias(data)
      Some(weights.b.t * pinv(weights.A) * x)
    } catch {
      case e: Exception => e.printStackTrace()
        None
    }
  }

  override def fit(data: Point): Unit = {
    val x: BreezeDenseVector[Double] = add_bias(data)
    try {
      weights += MatrixBias(x * x.t, data.asInstanceOf[LabeledPoint].label * x)
    } catch {
      case _: Exception =>
        if (weights == null) initialize_model(data)
        fit(data)
    }
  }

  override def fitLoss(data: Point): Double = {
    val loss: Double =
      Math.pow(data.asInstanceOf[LabeledPoint].label - predict(data).get, 2) +
        lambda * Math.pow(weights.FrobeniusNorm, 2)
    fit(data: Point)
    loss
  }

  override def score(test_set: ListBuffer[Point]): Double =
    Scores.RMSE(test_set.asInstanceOf[ListBuffer[LabeledPoint]], this)

  private def model_init(n: Int): MatrixBias = {
    MatrixBias(lambda * diag(BreezeDenseVector.fill(n) {0.0}),
      BreezeDenseVector.fill(n) {0.0}
    )
  }

  private def add_bias(data: Point): BreezeDenseVector[Double] = {
    BreezeDenseVector.vertcat(
      data.getNumericVector.asBreeze.asInstanceOf[BreezeDenseVector[Double]],
      BreezeDenseVector.ones(1))
  }

  override def getParameters: Option[LearningParameters] = Option(weights)

  override def setParameters(params: LearningParameters): Learner = {
    assert(params.isInstanceOf[MatrixBias])
    weights = params.asInstanceOf[MatrixBias]
    this
  }

  def setLambda(lambda: Double): ORR = {
    this.lambda = lambda
    this
  }

  override def setParametersFromMap(parameterMap: mutable.Map[String, AnyRef]): Learner = {
    for ((parameter, value) <- parameterMap) {
      parameter match {
        case "A" =>
          try {
            val new_weights = BreezeDenseVector[Double](
              value.asInstanceOf[java.util.List[Double]].asScala.toArray
            ).toDenseMatrix
            if (weights == null || weights.A.size == new_weights.size)
              weights.A = new_weights
            else
              throw new RuntimeException("Invalid size of new A matrix for the ORR regressor.")
          } catch {
            case e: Exception =>
              println("Error while trying to update the matrix A of ORR regressor.")
              e.printStackTrace()
          }
        case "b" =>
          try {
            val new_bias = BreezeDenseVector[Double](value.asInstanceOf[java.util.List[Double]].asScala.toArray)
            if (weights == null || weights.b.size == new_bias.size)
              weights.b = new_bias
            else
              throw new RuntimeException("Invalid size of new b vector for the ORR regressor.")
          } catch {
            case e: Exception =>
              println("Error while trying to update the intercept flag of ORR regressor.")
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
        case "lambda" =>
          try {
            setLambda(value.asInstanceOf[Double])
          } catch {
            case e: Exception =>
              println("Error while trying to update the epsilon hyperparameter of PA regressor.")
              e.printStackTrace()
          }
        case _ =>
      }
    }
    this
  }

  override def toString: String = s"ORR ${this.hashCode}"

  override def generatePOJOLearner: LearnerPOJO = {
    new LearnerPOJO("ORR",
      Map[String, AnyRef](("lambda", lambda.asInstanceOf[AnyRef])).asJava,
      Map[String, AnyRef](
        ("a", if(weights == null) null else weights.A.data.asInstanceOf[AnyRef]),
        ("b", if(weights == null) null else weights.b.data.asInstanceOf[AnyRef])
      ).asJava,
      null
    )
  }

  override def generateParameters: ParameterDescriptor => LearningParameters = new MatrixBias().generateParameters

  override def getSerializedParams: (LearningParameters , Array[_]) => java.io.Serializable =
    new MatrixBias().generateSerializedParams

}
