package mlAPI.safezones

import mlAPI.parameters.VectoredParameters

/** The basic trait of a safe zone function. */
trait SafeZone {
  def zeta(globalModel: VectoredParameters, model: VectoredParameters): Double
  def newRoundZeta(): Double
}
