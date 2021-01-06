package mlAPI.safezones

import mlAPI.parameters.{BreezeParameters, DLParams, VectoredParameters}

/**
 * The variance safe zone function of the FGM distributed learning protocol.
 *
 * @param threshold The threshold of the FMG safe zone function.
 */
case class VarianceSafeZone(private var threshold: Double = 0.0008)
  extends SafeZone {

  var sqrtThreshold: Double = scala.math.sqrt(threshold)

  /** Calculation of the Zeta safe zone function. */
  override def zeta(globalModel: VectoredParameters, model: VectoredParameters): Double =
    sqrtThreshold - (globalModel - model).asInstanceOf[VectoredParameters].FrobeniusNorm

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
