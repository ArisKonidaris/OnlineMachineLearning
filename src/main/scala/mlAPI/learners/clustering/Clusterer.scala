package mlAPI.learners.clustering

import mlAPI.learners.Learner
import mlAPI.math.LearningPoint

trait Clusterer extends Learner {

  def distribution(data: LearningPoint): Array[Double]

}
