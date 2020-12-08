package mlAPI.learners.classification.trees.stats

import mlAPI.learners.classification.trees.serializable.stats.GaussianDescriptor

/**
 * A Gaussian distribution.
 *
 * @param mean      The mean value of the Gaussian distribution.
 * @param d_squared The standard deviation of the attribute times the number of values needed to
 *                  built the Gaussian distribution minus 1.
 * @param count     The number of values needed to built the Gaussian distribution.
 */
case class GaussianDistribution(private var mean: Double = 0D,
                                private var d_squared: Double = 0D,
                                private var count: Long = 0L) {


  //////////////////////////////////////////////////// Getters /////////////////////////////////////////////////////////

  def getCount: Long = count

  def getMean: Double = mean

  def getVariance: Double = if (count > 1L) Math.pow(1.0 * (count - 1L), -1) * d_squared else 0.0D

  def getStandardDeviation: Double = Math.sqrt(getVariance)

  //////////////////////////////////////////////////// Methods /////////////////////////////////////////////////////////

  /**
   * A method for calculating the probability density of the Gaussian distribution on a given value.
   *
   * @param value A value to calculate the probability density on.
   * @return The probability density of the Gaussian distribution on a given value.
   */
  def getProbabilityDensity(value: Double): Double = {
    if (count > 0.0D) {
      val stdDev: Double = getStandardDeviation
      if (stdDev > 0.0D) {
        val diff = value - mean
        1.0D / (GaussianDistribution.NORMAL_CONSTANT * stdDev) * Math.exp(-(diff * diff / (2.0D * stdDev * stdDev)))
      } else if (value == mean) 1.0D else 0.0D
    } else 0.0D
  }

  def getWeights(value: Double): (Double, Double, Double) = {
    val equalToWeight: Double = getProbabilityDensity(value) * count
    val stdDev: Double = getStandardDeviation
    val lessThanWeight: Double = {
      if (stdDev > 0.0D)
        GaussianDistribution.normalProbability((value - mean) / stdDev) * count - equalToWeight
      else if (value < mean) count - equalToWeight else 0.0D
    }
    var greaterThanWeight: Double = count - equalToWeight - lessThanWeight
    if (greaterThanWeight < 0.0D) greaterThanWeight = 0.0D
    (lessThanWeight, equalToWeight, greaterThanWeight)
  }

  /**
   * Fit a value to the Gaussian distribution.
   *
   * @param value A value to fit into the Gaussian distribution.
   */
  def fit(value: Double): Unit = {
    count += 1
    val newMean: Double = mean + (1.0 / count) * (value - mean)
    d_squared += (value - newMean) * (value - mean)
    mean = newMean
  }

  /**
   * A method for converting a [[GaussianDistribution]] instance into a [[Serializable]] instance.
   *
   * @return A Serializable instance of a Gaussian distribution.
   */
  def serialize: GaussianDescriptor = new GaussianDescriptor(mean, d_squared, count)

  /**
   * A method for merging two Gaussian distributions.
   *
   * @param approx A Gaussian distribution to merge with this one.
   */
  def merge(approx: GaussianDistribution): Unit = {
    val n1: Long = count
    val n2: Long = approx.count
    val m1: Double = mean
    val m2: Double = approx.mean
    val v1: Double = d_squared / (1.0 * n1)
    val v2: Double = approx.d_squared / (1.0 * n2)
    count = n1 + n2
    mean = (m1 * n1 + m2 * n2) / (1.0 * (n1 + n2))
    d_squared = n1 * (v1 + Math.pow(m1 - mean, 2)) + n2 * (v2 + Math.pow(m2 - mean, 2))
  }

}

object GaussianDistribution {

  val NORMAL_CONSTANT: Double = Math.sqrt(6.283185307179586D)

  def p1evl(x: Double, coef: Array[Double], N: Int): Double = {
    var ans = x + coef(0)
    for (i <- 1 until N) {
      ans = ans * x + coef(i)
    }
    ans
  }

  def polevl(x: Double, coef: Array[Double], N: Int): Double = {
    var ans = coef(0)
    for (i <- 1 to N) {
      ans = ans * x + coef(i)
    }
    ans
  }

  def errorFunctionComplemented(a: Double): Double = {
    val P = Array[Double](2.461969814735305E-10D, 0.5641895648310689D, 7.463210564422699D, 48.63719709856814D, 196.5208329560771D, 526.4451949954773D, 934.5285271719576D, 1027.5518868951572D, 557.5353353693994D)
    val Q = Array[Double](13.228195115474499D, 86.70721408859897D, 354.9377788878199D, 975.7085017432055D, 1823.9091668790973D, 2246.3376081871097D, 1656.6630919416134D, 557.5353408177277D)
    val R = Array[Double](0.5641895835477551D, 1.275366707599781D, 5.019050422511805D, 6.160210979930536D, 7.4097426995044895D, 2.9788666537210022D)
    val S = Array[Double](2.2605286322011726D, 9.396035249380015D, 12.048953980809666D, 17.08144507475659D, 9.608968090632859D, 3.369076451000815D)
    var x = .0
    if (a < 0.0D) x = -a
    else x = a
    if (x < 1.0D) 1.0D - errorFunction(a)
    else {
      var z = -a * a
      if (z < -709.782712893384D) if (a < 0.0D) 2.0D
      else 0.0D
      else {
        z = Math.exp(z)
        var p = .0
        var q = .0
        if (x < 8.0D) {
          p = polevl(x, P, 8)
          q = p1evl(x, Q, 8)
        }
        else {
          p = polevl(x, R, 5)
          q = p1evl(x, S, 6)
        }
        var y = z * p / q
        if (a < 0.0D) y = 2.0D - y
        if (y == 0.0D) if (a < 0.0D) 2.0D
        else 0.0D
        else y
      }
    }
  }

  def errorFunction(x: Double): Double = {
    val T = Array[Double](9.604973739870516D, 90.02601972038427D, 2232.005345946843D, 7003.325141128051D, 55592.30130103949D)
    val U = Array[Double](33.56171416475031D, 521.3579497801527D, 4594.323829709801D, 22629.000061389095D, 49267.39426086359D)
    if (Math.abs(x) > 1.0D) 1.0D - errorFunctionComplemented(x)
    else {
      val z = x * x
      val y = x * polevl(z, T, 4) / p1evl(z, U, 5)
      y
    }
  }

  def normalProbability(a: Double): Double = {
    val x = a * 0.7071067811865476D
    val z = Math.abs(x)
    var y = .0
    if (z < 0.7071067811865476D) y = 0.5D + 0.5D * errorFunction(x)
    else {
      y = 0.5D * errorFunctionComplemented(z)
      if (x > 0.0D) y = 1.0D - y
    }
    y
  }

  def getSize = 24

  def deserialize(descriptor: GaussianDescriptor): GaussianDistribution =
    GaussianDistribution(descriptor.getMean, descriptor.getDSquared, descriptor.getCount)

}
