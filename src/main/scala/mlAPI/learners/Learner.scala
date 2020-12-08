package mlAPI.learners

import ControlAPI.LearnerPOJO
import mlAPI.math.Point
import mlAPI.parameters.{LearningParameters, ParameterDescriptor, WithParams}

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

/**
 * Contains the necessary methods needed by the workers/slave node
 * to train on it's local incoming data stream.
 */
trait Learner extends Serializable with WithParams {

  // An immutable variable that determines if the Learner can be parallelized.
  protected val parallelizable: Boolean

  // An integer indicating the update complexity of the Learner on a single data point.
  protected var update_complexity: Int = _

  // ===================================== Getters ================================================

  def getParameters: Option[LearningParameters]

  def getUpdateComplexity: Int = update_complexity

  // ===================================== Setters ================================================

  def setParameters(params: LearningParameters): Learner

  def setUpdateComplexity(update_complexity: Int): Unit = this.update_complexity = update_complexity

  // ==================================== Main methods =============================================

  override def setHyperParametersFromMap(hyperParameterMap: mutable.Map[String, AnyRef]): Learner = this

  override def addHyperParameter(key: String, value: AnyRef): Learner = this

  override def removeHyperParameter(key: String, value: AnyRef): Learner = this

  override def setParametersFromMap(parameterMap: mutable.Map[String, AnyRef]): Learner = this

  override def addParameter(key: String, value: AnyRef): Learner = this

  override def removeParameter(key: String, value: AnyRef): Learner = this

  def initialize_model(): Unit = ()

  def initialize_model(data: Point): Unit = ()

  def predict(data: Point): Option[Double]

  def fit(data: Point): Unit

  def fit(batch: ListBuffer[Point]): Unit

  def fitLoss(data: Point): Double

  def fitLoss(batch: ListBuffer[Point]): Double

  def score(test_set: ListBuffer[Point]): Double

  def generateParameters: ParameterDescriptor => LearningParameters

  def getSerializedParams: (LearningParameters, Array[_]) => java.io.Serializable

  def generatePOJOLearner: LearnerPOJO

}
