package mlAPI.learners

import ControlAPI.LearnerPOJO
import mlAPI.math.Point
import mlAPI.parameters.{LearningParameters, ParameterDescriptor, SerializedParameters, WithParams}

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

/**
 * Contains the necessary methods needed by the workers/slave node
 * to train on it's local incoming data stream.
 */
trait Learner extends Serializable with WithParams {

  // The size of the mini batch.
  protected var miniBatchSize: Int

  // An immutable variable that determines if the Learner can be parallelized.
  protected val parallelizable: Boolean

  // An integer indicating the update complexity of the Learner on a single data point.
  protected var updateComplexity: Int = _

  // ===================================== Getters ================================================

  def getMiniBatchSize: Int = miniBatchSize

  def isParallelizable: Boolean = parallelizable

  def getUpdateComplexity: Int = updateComplexity

  def getParameters: Option[LearningParameters]

  // ===================================== Setters ================================================

  def setMiniBatchSize(miniBatchSize: Int): Learner = {
    this.miniBatchSize = miniBatchSize
    this
  }

  def setUpdateComplexity(update_complexity: Int): Learner = {
    this.updateComplexity = update_complexity
    this
  }

  def setParameters(params: LearningParameters): Learner

  // ==================================== Main methods =============================================

  override def setHyperParametersFromMap(hyperParameterMap: mutable.Map[String, AnyRef]): Learner = this

  override def addHyperParameter(key: String, value: AnyRef): Learner = this

  override def removeHyperParameter(key: String, value: AnyRef): Learner = this

  override def setParametersFromMap(parameterMap: mutable.Map[String, AnyRef]): Learner = this

  override def setStructureFromMap(structureMap: mutable.Map[String, AnyRef]): Learner = this

  override def addParameter(key: String, value: AnyRef): Learner = this

  override def removeParameter(key: String, value: AnyRef): Learner = this

  def initializeModel(): Learner = this

  def initializeModel(data: Point): Learner = this

  def predict(data: Point): Option[Double]

  def predict(batch: ListBuffer[Point]): Array[Option[Double]]

  def fit(data: Point): Unit

  def fit(batch: ListBuffer[Point]): Unit

  def fitLoss(data: Point): Double

  def fitLoss(batch: ListBuffer[Point]): Double

  def score(test_set: ListBuffer[Point]): Double

  def generateParameters: ParameterDescriptor => LearningParameters

  def getSerializedParams: (LearningParameters, Array[_]) => SerializedParameters

  def generatePOJOLearner: LearnerPOJO

}
