package mlAPI.preprocessing

import mlAPI.math.LearningPoint

import scala.collection.mutable.ListBuffer

abstract class learningPreprocessor extends Preprocessor {

  protected var learnable: Boolean = true

  def init(point: LearningPoint): Unit

  def isLearning: Boolean = learnable

  def freezeLearning(): Unit = learnable = false

  def enableLearning(): Unit = learnable = true

  def fit(point: LearningPoint): Unit

  def fit(dataSet: ListBuffer[LearningPoint]): Unit
}
