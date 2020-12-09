package mlAPI.safezones

import mlAPI.parameters.BreezeParameters

/**
 * The variance safe zone function of the FGM distributed learning protocol.
 *
 * @param threshold The threshold of the FMG safe zone function.
 */
case class VarianceSafeZone(private var threshold: Double = 0.008)
  extends SafeZone {

  var sqrtThreshold: Double = scala.math.sqrt(threshold)

  /** Calculation of the Zeta safe zone function. */
  override def zeta(globalModel: BreezeParameters, model: BreezeParameters): Double = {
    try {
      assert(globalModel != null)
      assert(model != null)
      sqrtThreshold - scala.math.sqrt(breeze.linalg.norm(globalModel.flatten - model.flatten))
    } catch {
      case e: Throwable =>
        e.printStackTrace()
        throw new RuntimeException("Check this.")
    }

  }

  override def newRoundZeta(): Double = sqrtThreshold

  def getThreshold: Double = {
    val value: Double = threshold
    value
  }

  def setThreshold(threshold: Double): Unit = {
    this.threshold = threshold
    sqrtThreshold = scala.math.sqrt(threshold)
  }

}
