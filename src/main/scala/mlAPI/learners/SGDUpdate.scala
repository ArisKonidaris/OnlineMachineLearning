package mlAPI.learners

/**
 * A trait for learners that use an SGD update rule.
 */
trait SGDUpdate {

  var learningRate: Double

  def getLearningRate: Double = learningRate

  def setLearningRate(learningRate: Double): Unit = this.learningRate = learningRate

}
