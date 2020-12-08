package mlAPI.learners.classification.trees.stats

import mlAPI.learners.classification.trees.serializable.stats.{DiscreteStatisticsDescriptor, TargetCountersDescriptor}
import mlAPI.learners.classification.trees.serializable.stats.{ValuesDescriptor, DiscreteAttributeDescriptor}
import mlAPI.math.Vector

import scala.collection.mutable.{ListBuffer, HashMap => Stats}

/**
 * A set of counters holding the sufficient stats for each attribute.
 * There is one counter per class per value per attribute.
 */
case class DiscreteStatistics() extends Statistics {

  private val statistics: Stats[Int, Stats[Int, Stats[Int, Long]]] = Stats[Int, Stats[Int, Stats[Int, Long]]]()
  private var targets: ListBuffer[Int] = ListBuffer[Int]()
  private var dropped: ListBuffer[Int] = ListBuffer[Int]()

  //////////////////////////////////////////////////// Getters /////////////////////////////////////////////////////////

  def getStatistics: Stats[Int, Stats[Int, Stats[Int, Long]]] = statistics

  def getTargets: ListBuffer[Int] = targets

  def getDropped: ListBuffer[Int] = dropped

  //////////////////////////////////////////////////// Methods /////////////////////////////////////////////////////////

  override def mergeStats(stats: Statistics): Unit = {
    require(stats.isInstanceOf[DiscreteStatistics])
    super.mergeStats(stats)
    val st = stats.asInstanceOf[DiscreteStatistics]
    for (target <- st.targets) if (!targets.contains(target)) targets += target
    for (droppedAttribute <- st.dropped) if (!dropped.contains(droppedAttribute)) dropAttribute(droppedAttribute)
    for ((attr, valueStats) <- st.statistics) {
      if (!statistics.contains(attr)) {
        statistics.put(attr, valueStats)
      } else {
        val s = statistics(attr)
        for ((value, targetCounts) <- valueStats) {
          if (!s.contains(value)) {
            s.put(value, targetCounts)
          } else {
            val v = s(value)
            for ((target, counts) <- targetCounts) {
              if (!v.contains(target)) v.put(target, counts) else v(target) += counts
            }
          }
        }
      }
    }
  }

  override def updateStats(point: Vector, target: Int): Unit = {
    super.updateStats(point, target)
    if (isActive)
      for (attribute <- 0 until point.size)
        if (!dropped.contains(attribute))
          updateStatistics(attribute, point(attribute).toInt, target)
  }

  def updateStatistics(attribute: Int, value: Int, target: Int): Unit = {
    if (!statistics.contains(attribute)) statistics.put(attribute, Stats[Int, Stats[Int, Long]]())
    if (!statistics(attribute).contains(value)) statistics(attribute).put(value, Stats[Int, Long]())
    val counters: Stats[Int, Long] = statistics(attribute)(value)
    if (!counters.contains(target)) counters.put(target, 1L) else counters(target) += 1L
    if (!targets.contains(target)) {
      targets += target
      for ((attr: Int, valueCounts: Stats[Int, Stats[Int, Long]]) <- statistics) {
        for ((v: Int, counts: Stats[Int, Long]) <- valueCounts) {
          if (((attr == attribute) && (v != value)) || attr != attribute) {
            assert(!counts.keySet.contains(target))
            counts.put(target, 0L)
          }
        }
      }
    }
  }

  override def dropAttribute(attribute: Int): Unit = {
    if (statistics.contains(attribute)) {
      statistics.drop(attribute)
      dropped += attribute
    }
  }

  override def getClassDistributions: Array[(Int, Long)] = {
    (
      for ((_: Int, classCounts: Stats[Int, Long]) <- statistics.head._2)
        yield
          classCounts.toArray.sortBy(x => x._1)
      ).toArray.reduce((x, y) => x.zip(y).map { case (z, k) => (z._1, z._2 + k._2) })
  }

  override def serialize: DiscreteStatisticsDescriptor = {
    val serializedStats = serializeStats
    val discreteAttributeStats: DiscreteAttributeDescriptor = {
      val attributes = ListBuffer[Int]()
      val counters = ListBuffer[ValuesDescriptor]()
      statistics.foreach(x => {
        attributes += x._1
        val values = ListBuffer[Int]()
        val targetCounters = ListBuffer[TargetCountersDescriptor]()
        x._2.foreach(y => {
          values += y._1
          val targets = ListBuffer[Int]()
          val counters = ListBuffer[Long]()
          y._2.foreach(z => {
            targets += z._1
            counters += z._2
          })
          targetCounters += new TargetCountersDescriptor(targets.toArray, counters.toArray)
        })
        counters += new ValuesDescriptor(values.toArray, targetCounters.toArray)
      })
      new DiscreteAttributeDescriptor(attributes.toArray, counters.toArray)
    }
    new DiscreteStatisticsDescriptor(
      serializedStats._1,
      serializedStats._2,
      serializedStats._3,
      serializedStats._4,
      serializedStats._5,
      serializedStats._6,
      discreteAttributeStats,
      targets.toArray,
      dropped.toArray
    )
  }

  def entropy(counts: Array[Double]): (Double, Double) = {
    val total: Double = counts.sum
    (
      (for (count: Double <- counts) yield {
        val fraction: Double = count / total
        -fraction * (Math.log10(fraction) / Math.log10(2))
      }).sum,
      total
    )
  }

  def addArrays(arrays: Array[Array[(Int, Double)]]): Array[(Int, Double)] =
    arrays.reduce((x, y) => x.zip(y).map { case (z, k) => (z._1, z._2 + k._2) })

  override def bestSplits: Array[(Int, Double, Double, Array[Array[(Int, Double)]])] = {

    (for ((attr, stats) <- statistics) yield {

      val st: Array[(Int, Array[(Int, Double)])] = stats.toArray.sortBy(x => x._1)
        .map(x => (x._1, x._2.toArray.sortBy(y => y._1).map(z => (z._1, 1.0 * z._2))))

      val valueSplit: (Int, Double, Array[Array[(Int, Double)]]) =
        (for (index <- 0 to st.length - 2) yield {
          val ar1: Array[(Int, Double)] = addArrays(st.slice(0, index + 1).map(x => x._2))
          val ar2: Array[(Int, Double)] = addArrays(st.slice(index + 1, st.length).map(x => x._2))
          val (e1, e2) = (entropy(ar1.map(x => x._2)), entropy(ar2.map(x => x._2)))
          val total: Double = e1._2 + e2._2
          (st(index)._1, (e1._2 / total) * e1._1 + (e2._2 / total) * e2._1, Array(ar1, ar2))
        }).toArray.minBy(x => x._2)

      (attr, valueSplit._1.toDouble, valueSplit._2, valueSplit._3)

    }).toArray.sortBy(x => x._3)

  }

  override def clear(): Unit = {
    statistics.clear()
    targets.clear()
    dropped.clear()
  }

  override def createStats: Statistics = DiscreteStatistics()

  override def generateStats: Statistics = {

    // Create the new Statistics instance.
    val new_statistics_instance: DiscreteStatistics = DiscreteStatistics()

    // Create the new sufficient statistics counters.
    for ((attr: Int, stats: Stats[Int, Stats[Int, Long]]) <- statistics) {
      val new_stats = Stats[Int, Stats[Int, Long]]()
      for ((value: Int, class_counts: Stats[Int, Long]) <- stats) {
        val new_class_counts: Stats[Int, Long] = Stats[Int, Long]()
        for ((target, counts) <- class_counts) new_class_counts.put(target, counts)
        new_stats.put(value, new_class_counts)
      }
      new_statistics_instance.getStatistics.put(attr, new_stats)
    }

    // Create the target and dropped attribute lists.
    for (target <- targets) new_statistics_instance.getTargets += target
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
        val classDist = getClassDistributions
        val total: Long = getClassDistributions.map(x => x._2).sum
        val counts: Array[(Int, Double)] = (
          for (i <- 0 until point.size)
            yield
              statistics(i)(point(i).toInt).toArray.map(x => (x._1, x._2.toDouble)).sortBy(x => x._1)
          ).toArray.reduce((x, y) => x.zip(y).map { case (z, k) => (z._1, z._2 * k._2) })
        classDist.map(x => (x._1, x._2 / total)).zip(counts).map(x => (x._1._1, 1.0 * x._1._2 * x._2._2)).maxBy(x => x._2)
      } else super.predict(point)
    } else throw new RuntimeException("No such prediction method.")
  }

  override def getSize: Int = {
    super.getSize + targets.length * 4 + dropped.length * 4 + {
      if (statistics.isEmpty)
        0
      else
        (for ((_: Int, stats: Stats[Int, Stats[Int, Long]]) <- statistics) yield {
          4 + (for ((_: Int, targetCounters: Stats[Int, Long]) <- stats) yield {
            4 + targetCounters.size * 12
          }).sum
        }).sum
    }
  }

}
