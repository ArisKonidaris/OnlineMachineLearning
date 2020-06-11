package mlAPI.learners.clustering

import mlAPI.learners.Learner
import mlAPI.math.Point

trait Clusterer extends Learner {

  def distribution(data: Point): Array[Double]

}
