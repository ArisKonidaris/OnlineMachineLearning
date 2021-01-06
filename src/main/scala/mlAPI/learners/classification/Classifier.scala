package mlAPI.learners.classification

import mlAPI.learners.Learner

trait Classifier extends Learner {

  protected var targetLabel: Double

  def setTargetLabel(targetLabel: Double): Learner = {
    this.targetLabel = targetLabel
    this
  }

  def getTargetLabel: Double = {
    val value: Double = targetLabel
    value
  }

}
