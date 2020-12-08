package mlAPI.learners.classification.trees.stats

import mlAPI.learners.classification.trees.serializable.stats.{AttributeGaussianDescriptor, DiscreteStatisticsDescriptor, NumericalStatisticsDescriptor, StatisticsDescriptor, TargetCountersDescriptor, ValuesDescriptor}
import mlAPI.math.Vector

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

/**
 * A trait for the basic statistics. THis trait contains the class distribution and various other variables for
 * majority based prediction.
 */
trait Statistics {

  var classDistribution: mutable.HashMap[Int, Double] = mutable.HashMap[Int, Double]() // The class distribution.

  var n_l: Double = 0D // The number of data points used to construct the class distribution.

  var max: Double = 0D // The votes of the majority class.

  var prediction: Int = 0 // The majority class.

  var isActive: Boolean = true // A boolean value determining if an instance of Statistics is Active or not.

  /**
   * A getter that returns the distribution of classes in an Array.
   *
   * @return An array of 2-tuples where the first element is the class and the second element is a counter.
   */
  def getClassDistributions: Array[(Int, Long)]

  /**
   * Setting the statistics by using a class Distribution.
   *
   * @param class_distribution Class distribution.
   * @return A [[Statistics]] instance.
   */
  def setStats(class_distribution: Array[(Int, Double)]): Statistics = {
    prediction = class_distribution.maxBy(x => x._2)._1
    n_l = class_distribution.foldLeft(0.0D)((acc, x: (Int, Double)) => {
      acc + x._2
    })
    classDistribution.clear()
    for ((target: Int, count: Double) <- class_distribution) classDistribution.put(target, count)
    this
  }

  /**
   * A method that merged two instances of [[Statistics]].
   *
   * @param stats A [[Statistics]] instance to merge with the current one.
   */
  def mergeStats(stats: Statistics): Unit = {
    n_l += stats.n_l
    for ((target, counter) <- stats.classDistribution)
      if (classDistribution.contains(target))
        classDistribution(target) += counter
      else
        classDistribution.put(target, counter)
    val m = classDistribution.toArray.maxBy(x => x._2)
    prediction = m._1
    max = m._2
    if (isActive && stats.isActive) isActive = true else isActive = false
  }

  /**
   * This method fits a labeled data point to the sufficient statistics.
   *
   * @param point  The feature vector.
   * @param target The class.
   */
  def updateStats(point: Vector, target: Int): Unit = {
    val count = {
      try {
        classDistribution(target) + 1
      } catch {
        case _: java.util.NoSuchElementException =>
          classDistribution.put(target, 0)
          1
      }
    }
    if (count > max) {
      max = count
      prediction = target
    }
    classDistribution(target) = count
    n_l += 1
  }

  /**
   * By calling this method yoy discard any kept statistics for a specific attribute and will be ingored
   * in future training of the statistics.
   *
   * @param attribute The attribute to discard and to stop keeping statistics on.
   */
  def dropAttribute(attribute: Int): Unit

  /**
   * This method calculates the best splitting points for each attribute and sorts them based on their Information Gain.
   *
   * @return The best splitting points for each attribute, sorted by their Information Gain.
   */
  def bestSplits: Array[(Int, Double, Double, Array[Array[(Int, Double)]])]

  /**
   * Clears the statistics. This method is called upon the deactivation of a leaf.
   */
  def clear(): Unit = {
    val stats: Array[(Int, Long)] = getClassDistributions
    val n: Long = stats.map(x => x._2).sum
    n_l -= n
    prediction = if (n > n_l) getClassDistributions.maxBy(x => x._2)._1 else classDistribution.maxBy(x => x._2)._1
    for ((target, count) <- stats) if (classDistribution.contains(target)) classDistribution(target) -= count
  }

  /**
   * A method for creating a [[Statistics]] instance.
   *
   * @return A subclass instance of [[Statistics]].
   */
  def createStats: Statistics

  /**
   * A method for generating a copy of this [[Statistics]] instance.
   *
   * @return A copy of this [[Statistics]] instance.
   */
  def generateStats: Statistics

  /**
   * A method for prediction.
   *
   * @param point  A feature vector to give a prediction on.
   * @param method The technique of prediction. Available methods are the "MajorityVote" and "NaiveBayes".
   * @return A 2-tuple where the first element is the predicted class and the second element is the confidence.
   */
  def predict(point: Vector, method: String = "MajorityVote"): (Int, Double) = {
    val stats: Array[(Int, Long)] = getClassDistributions
    val mx: (Int, Long) = stats.maxBy(x => x._2)
    (mx._1, 1.0 * mx._2 / stats.map(x => x._2).sum)
  }

  /**
   * A method to convert a [[Statistics]] instance into a serializable object.
   *
   * @return
   */
  def serialize: StatisticsDescriptor

  /**
   * Calculated the entropy based on the class distribution.
   *
   * @return The entropy of the class distribution.
   */
  def getEntropy: Double = {
    classDistribution.toArray.foldLeft(0.0D)((acc, x: (Int, Double)) => {
      val x_frac: Double = x._2 / n_l
      acc - x_frac * (Math.log(x_frac) / Math.log(2.0D))
    })
  }

  /**
   * A method called upon the deactivation of a leaf.
   */
  def deactivate(): Unit = {
    clear()
    isActive = false
  }

  /**
   * A method called upon the activation of a leaf.
   */
  def activate(): Unit = isActive = true

  /**
   * A helper method for serializing the sufficient statistics.
   *
   * @return Serializable statistics.
   */
  def serializeStats: (Array[Int], Array[Double], Double, Int, Double, Boolean) = {
    val targets = ListBuffer[Int]()
    val counters = ListBuffer[Double]()
    classDistribution.foreach(x => {
      targets += x._1
      counters += x._2
    })
    (targets.toArray, counters.toArray, max, prediction, n_l, isActive)
  }

  /**
   * A helper method for deserializing sufficient statistics.
   *
   * @param descriptor The serializable statistics.
   */
  def deserializeStats(descriptor: (Array[Int], Array[Double], Double, Int, Double, Boolean)): Unit = {
    classDistribution.clear()
    descriptor._1.zip(descriptor._2).foreach(x => classDistribution.put(x._1, x._2))
    max = descriptor._3
    prediction = descriptor._4
    n_l = descriptor._5
    isActive = descriptor._6
  }

  /**
   * Returns the byte size of the sufficient statistics.
   *
   * @return Byte size.
   */
  def getSize: Int = 21 + classDistribution.size * 12

}

object Statistics {
  def deserializeStats(descriptor: StatisticsDescriptor): Statistics = {

    import scala.collection.mutable.{HashMap => Stats}

    descriptor match {
      case numStats: NumericalStatisticsDescriptor =>
        val stats = NumericalStatistics(numStats.getSplits)
        stats.deserializeStats(
          (numStats.getClassCountersKeys,
            numStats.getClassCountersValues,
            numStats.getMax,
            numStats.getPrediction,
            numStats.getNl,
            numStats.getIsActive
          )
        )
        stats.getStatistics.clear()
        for (agd: AttributeGaussianDescriptor <- numStats.getAttributeNormalsDescriptor) {
          val normals = Stats[Int, GaussianDistribution]()
          agd.getTargets.zip(agd.getNormals.map(x => GaussianDistribution.deserialize(x)))
            .foreach(x => normals.put(x._1, x._2))
          stats.getStatistics.put(agd.getAttribute, (Range.deserialize(agd.getRange), normals))
        }
        for (dropped: Int <- numStats.dropped) stats.dropAttribute(dropped)
        stats
      case dStats: DiscreteStatisticsDescriptor =>
        val stats = DiscreteStatistics()
        stats.deserializeStats(
          (dStats.getClassCountersKeys,
            dStats.getClassCountersValues,
            dStats.getMax,
            dStats.getPrediction,
            dStats.getNl,
            dStats.getIsActive
          )
        )
        stats.getTargets.clear()
        dStats.getTargets.foreach(x => {
          stats.getTargets += x
        })
        stats.getDropped.clear()
        dStats.getDropped.foreach(x => {
          stats.getDropped += x
        })
        stats.getStatistics.clear()
        for ((attr: Int, vStats: ValuesDescriptor) <- dStats.getStats.getAttributes.zip(dStats.getStats.getCounters)) {
          val valueStatistics = Stats[Int, Stats[Int, Long]]()
          for ((value: Int, counters: TargetCountersDescriptor) <- vStats.getValues.zip(vStats.getTargetCounters)) {
            val targetCounters = Stats[Int, Long]()
            for ((target: Int, counter: Long) <- counters.getTargets.zip(counters.getCounters)) {
              targetCounters.put(target, counter)
            }
            valueStatistics.put(value, targetCounters)
          }
          stats.getStatistics.put(attr, valueStatistics)
        }
        stats
      case _ => throw new RuntimeException("Unknown statistics Serialization Scheme.")
    }
  }
}