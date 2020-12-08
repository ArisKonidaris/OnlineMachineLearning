package mlAPI.safezones

import mlAPI.parameters.BreezeParameters

/** The basic trait of a safe zone function. */
trait SafeZone {
  def zeta(globalModel: BreezeParameters, model: BreezeParameters): Double
  def newRoundZeta(): Double
}
