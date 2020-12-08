package mlAPI.learners.classification.trees.stats

import mlAPI.learners.classification.trees.serializable.stats.{AttributeGaussianDescriptor, GaussianDescriptor, NumericalStatisticsDescriptor}
import mlAPI.math.Vector
import scala.collection.mutable.{ListBuffer, HashMap => Stats}
import scala.annotation.tailrec

/**
 * An approximation for sufficient statistics of numerical (real valued) attributes.
 *
 * @param splits The number of splits to test upon each attribute.
 */
case class NumericalStatistics(private var splits: Int) extends Statistics {

  private val statistics: Stats[Int, (Range, Stats[Int, GaussianDistribution])] =
    Stats[Int, (Range, Stats[Int, GaussianDistribution])]()
  private var dropped: ListBuffer[Int] = ListBuffer[Int]()
  private var splitPoints: Array[Double] = (for (i <- 1 to splits) yield (1.0 * i) / (splits + 1)).toArray

  //////////////////////////////////////////////////// Getters /////////////////////////////////////////////////////////

  def getDropped: ListBuffer[Int] = dropped

  def getStatistics: Stats[Int, (Range, Stats[Int, GaussianDistribution])] = statistics

  override def getClassDistributions: Array[(Int, Long)] = {
    if (statistics.isEmpty)
      Array[(Int, Long)]()
    else
      (
        for (stats: (Int, GaussianDistribution) <- statistics.head._2._2) yield (stats._1, stats._2.getCount)
        ).toArray.sortBy(x => x._1)
  }

  override def getSize: Int = {
    super.getSize + splitPoints.length * 8 + dropped.length * 4 + {
      if (statistics.isEmpty)
        0
      else
        (for ((_: Int, stats: (Range, Stats[Int, GaussianDistribution])) <- statistics) yield {
          4 + Range.getSize + stats._2.size * (4 + GaussianDistribution.getSize)
        }).sum
    }
  }

  //////////////////////////////////////////////////// Methods /////////////////////////////////////////////////////////

  class weightStats {
    var weights1: ListBuffer[(Int, Double, Double)] = ListBuffer[(Int, Double, Double)]()
    var sum1: Double = 0D
    var weights2: ListBuffer[(Int, Double, Double)] = ListBuffer[(Int, Double, Double)]()
    var sum2: Double = 0D
    var total: Double = 0D
  }

  @tailrec
  private def createWeights(normals: Array[(Int, GaussianDistribution)],
                            weights: weightStats,
                            split: Double)
  : weightStats = {
    if (normals.length == 0) {
      assert(weights.weights1.length == weights.weights2.length)
      weights.weights1 = weights.weights1.map(x => (x._1, x._2, x._2 / weights.sum1))
      weights.weights2 = weights.weights2.map(x => (x._1, x._2, x._2 / weights.sum2))
      weights.total = weights.sum1 + weights.sum2
      weights
    } else {
      val w: (Double, Double, Double) = normals.head._2.getWeights(split)
      val w1: Double = w._1 + w._2
      weights.sum1 += w1
      weights.sum2 += w._3
      weights.weights1.append((normals.head._1, w1, 0D))
      weights.weights2.append((normals.head._1, w._3, 0D))
      createWeights(normals.tail, weights, split)
    }
  }


  override def mergeStats(stats: Statistics): Unit = {
    require(stats.isInstanceOf[NumericalStatistics])
    super.mergeStats(stats)
    val st = stats.asInstanceOf[NumericalStatistics]
    if (splitPoints.length < st.splitPoints.length) splitPoints = st.splitPoints
    for (droppedAttribute <- st.dropped) if (!dropped.contains(droppedAttribute)) dropAttribute(droppedAttribute)
    for ((attr, approx) <- st.statistics) {
      if (!statistics.contains(attr)) {
        statistics.put(attr, approx)
      } else {
        val s = statistics(attr)
        s._1.setLeftEnd(Math.min(s._1.getLeftEnd, approx._1.getLeftEnd))
        s._1.setRightEnd(Math.max(s._1.getRightEnd, approx._1.getRightEnd))
        for ((target, normal) <- approx._2)
          if (!s._2.contains(target))
            s._2.put(target, normal)
          else
            s._2(target).merge(normal)
      }
    }
  }

  def setSplits(splits: Int): Unit = {
    this.splits = splits
    splitPoints = (for (i <- 1 to splits) yield (1.0 * i) / (splits + 1)).toArray
  }

  override def updateStats(point: Vector, target: Int): Unit = {
    super.updateStats(point, target)
    if (isActive)
      for (attribute <- 0 until point.size)
        if (!dropped.contains(attribute))
          updateStatistics(attribute, point(attribute), target)
  }

  def updateStatistics(attribute: Int, value: Double, target: Int): Unit = {
    if (!statistics.contains(attribute))
      statistics.put(attribute, (new Range(), Stats[Int, GaussianDistribution]()))
    if (!statistics(attribute)._2.contains(target))
      statistics(attribute)._2.put(target, new GaussianDistribution())
    statistics(attribute)._1.update(value)
    statistics(attribute)._2(target).fit(value)
  }

  override def dropAttribute(attribute: Int): Unit = {
    if (statistics.contains(attribute)) statistics.drop(attribute)
    dropped += attribute
  }

  override def serialize: NumericalStatisticsDescriptor = {
    val serializedStats = serializeStats
    val attributeGaussianDescriptor = {
      (for ((attr: Int, stats: (Range, Stats[Int, GaussianDistribution])) <- statistics) yield {
        val targets = ListBuffer[Int]()
        val normals = ListBuffer[GaussianDescriptor]()
        stats._2.foreach(x => {
          targets += x._1
          normals += x._2.serialize
        })
        new AttributeGaussianDescriptor(attr, stats._1.serialize, targets.toArray, normals.toArray)
      }).toArray
    }
    new NumericalStatisticsDescriptor(
      serializedStats._1,
      serializedStats._2,
      serializedStats._3,
      serializedStats._4,
      serializedStats._5,
      serializedStats._6,
      attributeGaussianDescriptor,
      dropped.toArray,
      splits)
  }

  override def bestSplits: Array[(Int, Double, Double, Array[Array[(Int, Double)]])] = {

    val entropy = getEntropy
    (for ((attr, stats) <- statistics) yield {

      val candidateSplitPoints: Array[Double] = splitPoints
        .map(x => x * stats._1.getRange + stats._1.getLeftEnd)

      val (bestSplit, bestSplitEntropy, classDistributions) = {
        val candidateSplits = {
          for (split: Double <- candidateSplitPoints) yield {
            val weights: weightStats = createWeights(stats._2.toArray, new weightStats(), split)
            val IG: Double = {
              val w1: Double = weights.sum1 / weights.total
              val w2: Double = weights.sum2 / weights.total
              if ((w1 < 0.01) || (w2 < 0.01))
                Double.NegativeInfinity
              else {
                val ig = entropy -
                  (w1 * weights.weights1.map(x => -x._3 * (Math.log(x._3) / Math.log(2.0D))).sum +
                    w2 * weights.weights2.map(x => -x._3 * (Math.log(x._3) / Math.log(2.0D))).sum)
                if (ig.isNaN) Double.NegativeInfinity else ig
              }
            }
            (
              split,
              IG,
              Array(weights.weights1.toArray.map(x => (x._1, x._2)), weights.weights2.toArray.map(x => (x._1, x._2)))
            )
          }
        }
        candidateSplits.maxBy(x => x._2)
      }

      (attr, bestSplit, bestSplitEntropy, classDistributions)
    }).toArray.sortBy(x => -x._3)

  }

  override def clear(): Unit = {
    super.clear()
    statistics.clear()
    dropped.clear()
    splitPoints = Array[Double]()
  }

  override def createStats: Statistics = NumericalStatistics(splits)

  override def generateStats: Statistics = {

    // Create the new Statistics instance.
    val new_statistics_instance: NumericalStatistics = NumericalStatistics(splits)

    // Create the new Gaussian approximation.
    for ((attr: Int, stats: (Range, Stats[Int, GaussianDistribution])) <- statistics) {
      val new_stats: (Range, Stats[Int, GaussianDistribution]) = (stats._1.copy(), Stats[Int, GaussianDistribution]())
      for ((target: Int, gS: GaussianDistribution) <- stats._2) new_stats._2.put(target, gS.copy())
      new_statistics_instance.getStatistics.put(attr, new_stats)
    }

    // Create the dropped attribute list.
    for (dropped_attribute <- dropped) new_statistics_instance.getDropped += dropped_attribute

    // Copy the basic statistics.
    new_statistics_instance.n_l = n_l
    new_statistics_instance.max = max
    new_statistics_instance.prediction = prediction
    new_statistics_instance.isActive = isActive
    for ((target, counts) <- classDistribution) new_statistics_instance.classDistribution.put(target, counts)

    new_statistics_instance
  }

  override def predict(point: Vector, method: String = "MajorityVote"): (Int, Double) = {
    if (method.equals("MajorityVote"))
      super.predict(point)
    else if (method.equals("NaiveBayes")) {
      if (isActive) {
        val attrProb: Array[(Int, Double)] = {
          (for ((attr, stats) <- statistics) yield
            (for ((target: Int, gS) <- stats._2) yield
              (target, gS.getProbabilityDensity(point(attr)))
              ).toArray.sortBy(x => x._1)
            ).toArray
            .reduce((x: Array[(Int, Double)], y: Array[(Int, Double)]) =>
              x.zip(y).map { case (z: (Int, Double), k: (Int, Double)) => (z._1, z._2 * k._2) })
        }
        val pred = classDistribution.toArray.sortBy(x => x._1).zip(attrProb)
          .map { case (x,y) => (y._1, y._2 * (x._2 / (for (i <-statistics(0)._2) yield i._2.getCount).sum)) }
          .maxBy(x => x._2)
        (pred._1, pred._2)
      } else super.predict(point)
    } else throw new RuntimeException("No such prediction method.")
  }

  override def activate(): Unit = {
    super.activate()
    setSplits(splits)
  }

}
