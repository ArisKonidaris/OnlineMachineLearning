package mlAPI.protocols.dynamic

import ControlAPI.CountableSerial
import mlAPI.protocols.DoubleWrapper

case class ZetaValue(var zeta: Double, var phi: DoubleWrapper) extends ValueContainer[Double] with CountableSerial {

  def this() = this(0.0, null)

  def this(zeta: Double) = this(zeta, null)

  override def getValue: Double = zeta

  def getPhi: DoubleWrapper = phi

  override def setValue(value: Double): Unit = this.zeta = zeta

  def setPhi(phi: DoubleWrapper): Unit = this.phi = phi

  override def getSize: Int = 8 + { if (phi != null ) phi.getSize else 0 }
}