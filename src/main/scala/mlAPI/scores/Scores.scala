package mlAPI.scores

import mlAPI.learners.classification.Classifier
import mlAPI.learners.clustering.Clusterer
import mlAPI.learners.regression.Regressor
import mlAPI.math.{LabeledPoint, Point}

import scala.collection.mutable.ListBuffer

object Scores {

  def accuracy(testSet: ListBuffer[LabeledPoint], learner: Classifier): Double = {
    try {
      if (testSet.nonEmpty) {
        (for (test <- testSet) yield {
          val prediction: Double = learner.predict(test).get
          if (test.asInstanceOf[LabeledPoint].label == prediction) 1 else 0
        }).sum / (1.0 * testSet.length)
      } else 0.0
    } catch {
      case _: Throwable => 0.0
    }
  }

  def RMSE(testSet: ListBuffer[LabeledPoint], learner: Regressor): Double = {
    try {
      if (testSet.nonEmpty) {
        Math.sqrt(
          (for (test <- testSet) yield {
            learner.predict(test) match {
              case Some(pred) => Math.pow(test.asInstanceOf[LabeledPoint].label - pred, 2)
              case None => Double.MaxValue
            }
          }).sum / (1.0 * testSet.length)
        )
      } else Double.MaxValue
    } catch {
      case _: Throwable => Double.MaxValue
    }
  }

  def F1Score(testSet: ListBuffer[LabeledPoint], learner: Classifier): Double = {

    def frac(a: Int, b: Int): Double = {
      val denom: Int = a + b
      if (denom == 0) 0D else (1.0 * a) / denom
    }

    var truePositive: Int = 0
    var falsePositive: Int = 0
    var trueNegative: Int = 0
    var falseNegative: Int = 0
    try {
      if (testSet.nonEmpty) {
        for (point: LabeledPoint <- testSet) {
          val true_label = if (point.label > 0.0) 1D else -1D
          val prediction: Double = learner.predict(point).get
          if (true_label > 0.0 && prediction > 0.0)
            truePositive += 1
          else if (true_label > 0.0 && prediction < 0.0)
            falseNegative += 1
          else if (true_label < 0.0 && prediction > 0.0)
            falsePositive += 1
          else if (true_label < 0.0 && prediction < 0.0)
            trueNegative += 1
        }
        val precision: Double = frac(truePositive, falsePositive)
        val recall: Double = frac(truePositive, falseNegative)
        println("################################")
        println("truePositive: " + truePositive)
        println("falseNegative: " + falseNegative)
        println("falsePositive: " + falsePositive)
        println("trueNegative: " + trueNegative)
        println("precision: " + precision)
        println("recall: " + recall)
        if (precision + recall == 0)
          0D
        else
          2D * (precision * recall) / (precision + recall)
      } else 0.0
    } catch {
      case _: Throwable => 0.0
    }
  }

  def inertia(testSet: ListBuffer[Point], learner: Clusterer): Double = {
    if (testSet.nonEmpty) {
      @scala.annotation.tailrec
      def accumulateLoss(index: Int, loss: Double): Double = {
        require(index >= 0 && index <= testSet.length - 1)
        val dist: Array[Double] = learner.distribution(testSet(index))
        if (!dist.isEmpty) {
          val testLoss: Double = Math.pow(dist.min, 2)
          if (index == testSet.length - 1)
            loss + testLoss
           else
            accumulateLoss(index + 1, loss + testLoss)
        } else Double.MaxValue
      }
      accumulateLoss(0, 0.0)
    } else Double.MaxValue
  }

}
