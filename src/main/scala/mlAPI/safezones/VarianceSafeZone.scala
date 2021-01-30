package mlAPI.safezones

import mlAPI.parameters.VectoredParameters

/**
 * The variance safe zone function of the FGM distributed learning protocol.
 *
 * @param threshold The threshold of the FMG safe zone function.
 */
case class VarianceSafeZone(private var threshold: Double = 0.0008) extends SafeZone {

  var sqrtThreshold: Double = scala.math.sqrt(threshold)

//  println(threshold + " " + sqrtThreshold)

  /** Calculation of the Zeta safe zone function. */
  override def zeta(globalModel: VectoredParameters, model: VectoredParameters): Double =
    sqrtThreshold - (model - globalModel).asInstanceOf[VectoredParameters].frobeniusNorm

  override def newRoundZeta(): Double = sqrtThreshold

  def getThreshold: Double = threshold

  def setThreshold(threshold: Double): Unit = {
    this.threshold = threshold
    sqrtThreshold = scala.math.sqrt(threshold)
  }

}
