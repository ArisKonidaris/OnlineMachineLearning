package mlAPI.learners

import mlAPI.math.Breeze._
import mlAPI.math.Point
import mlAPI.parameters.{LearningParameters, ParameterDescriptor, SerializedParameters, SerializedVectoredParameters, VectorBias}
import breeze.linalg.{DenseVector => BreezeDenseVector}

import java.util
import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ListBuffer

abstract class PassiveAggressiveLearners extends Learner {

  override protected var miniBatchSize: Int = 1
  override protected val parallelizable: Boolean = true
  protected var updateType: String = "PA-II"
  protected var C: Double = 0.01
  protected var weights: VectorBias = _

  override def initializeModel(data: Point): Learner = {
    weights = VectorBias(BreezeDenseVector.zeros[Double](data.getNumericVector.size), 0.0)
    this
  }

  protected def predictWithMargin(data: Point): Option[Double] = {
    try {
      Some((data.getNumericVector.asBreeze dot weights.weights) + weights.intercept)
    } catch {
      case _: Throwable => None
    }
  }

  override def fit(batch: ListBuffer[Point]): Unit = {
    fitLoss(batch)
    ()
  }

  override def fitLoss(batch: ListBuffer[Point]): Double = (for (point <- batch) yield fitLoss(point)).sum

  protected def LagrangeMultiplier(loss: Double, data: Point): Double = {
    updateType match {
      case "STANDARD" => loss / (1.0 + ((data.getNumericVector dot data.getNumericVector) + 1.0))
      case "PA-I" => Math.min(C, loss / ((data.getNumericVector dot data.getNumericVector) + 1.0))
      case "PA-II" => loss / (((data.getNumericVector dot data.getNumericVector) + 1.0) + 1.0 / (2.0 * C))
    }
  }

  protected def checkParameters(data: Point): Unit = {
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

  override def getParameters: Option[LearningParameters] = Option(weights)

  override def setParameters(params: LearningParameters): Learner = {
    assert(params.isInstanceOf[VectorBias])
    weights = params.asInstanceOf[VectorBias]
    this
  }

  def setC(c: Double): PassiveAggressiveLearners = {
    this.C = c
    this
  }

  def setType(updateType: String): PassiveAggressiveLearners = {
    this.updateType = updateType
    this
  }

  override def predict(batch: ListBuffer[Point]): Array[Option[Double]] = {
    val predictions: ListBuffer[Option[Double]] = ListBuffer[Option[Double]]()
    for (point <- batch)
      predictions append predict(point)
    predictions.toArray
  }

  override def setParametersFromMap(parameterMap: mutable.Map[String, AnyRef]): Learner = {
    for ((parameter, value) <- parameterMap) {
      parameter match {
        case "a" =>
          try {
            val new_weights = BreezeDenseVector[Double](value.asInstanceOf[util.List[Double]].asScala.toArray)
            if (weights == null || weights.weights.size == new_weights.size)
              weights.weights = new_weights
            else
              throw new RuntimeException("Invalid size of new weight vector for the PA classifier.")
          } catch {
            case e: Exception =>
              println("Error while trying to update the weights of the PA classifier.")
              e.printStackTrace()
          }
        case "b" =>
          try {
            weights.intercept = value.asInstanceOf[Double]
          } catch {
            case e: Exception =>
              println("Error while trying to update the intercept of the PA classifier.")
              e.printStackTrace()
          }
        case _ =>
      }
    }
    this
  }

  override def generateParameters: ParameterDescriptor => LearningParameters = new VectorBias().generateParameters

  override def getSerializedParams: (LearningParameters, Array[_]) => SerializedParameters =
    new VectorBias().generateSerializedParams

}
