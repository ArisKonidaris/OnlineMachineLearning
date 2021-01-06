package mlAPI.protocols.fgm

import ControlAPI.CountableSerial

case class ZetaValue(var zeta: Double) extends ValueContainer[Double] with CountableSerial {

  def this() = this(0.0)

  override def getValue: Double = zeta

  override def setValue(value: Double): Unit = this.zeta = zeta

  override def getSize: Int = 8
}
